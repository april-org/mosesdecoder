#include "ConstrainedDecodingDeseg.h"
#include "moses/Hypothesis.h"
#include "moses/Manager.h"
#include "moses/ChartHypothesis.h"
#include "moses/ChartManager.h"
#include "moses/StaticData.h"
#include "moses/InputFileStream.h"
#include "moses/Util.h"
#include "util/exception.hh"

using namespace std;

namespace Moses
{
ConstrainedDecodingDesegState::ConstrainedDecodingDesegState(const Hypothesis &hypo)
{
  hypo.GetOutputPhrase(m_outputPhrase);
}

ConstrainedDecodingDesegState::ConstrainedDecodingDesegState(const ChartHypothesis &hypo)
{
  hypo.GetOutputPhrase(m_outputPhrase);
}

int ConstrainedDecodingDesegState::Compare(const FFState& other) const
{
  const ConstrainedDecodingDesegState &otherFF = static_cast<const ConstrainedDecodingDesegState&>(other);
  int ret =     m_outputPhrase.Compare(otherFF.m_outputPhrase);
  return ret;
}

//////////////////////////////////////////////////////////////////
ConstrainedDecodingDeseg::ConstrainedDecodingDeseg(const std::string &line)
  :StatefulFeatureFunction(1, line)
  ,m_maxUnknowns(0)
  ,m_negate(false)
  ,m_soft(false)
{
  m_tuneable = false;
  ReadParameters();
}

void ConstrainedDecodingDeseg::Load()
{
  const StaticData &staticData = StaticData::Instance();
  bool addBeginEndWord = (staticData.options().search.algo == CYKPlus) || (staticData.options().search.algo == ChartIncremental);

  for(size_t i = 0; i < m_paths.size(); ++i) {
    InputFileStream constraintFile(m_paths[i]);
    std::string line;
    long sentenceID = staticData.GetStartTranslationId() - 1;
    while (getline(constraintFile, line)) {
      vector<string> vecStr = Tokenize(line, "\t");

      Phrase phrase(0);
      if (vecStr.size() == 1) {
        sentenceID++;
        // phrase.CreateFromString(Output, staticData.GetOutputFactorOrder(), vecStr[0], staticData.GetFactorDelimiter(), NULL);
        phrase.CreateFromString(Output, staticData.GetOutputFactorOrder(), vecStr[0], NULL);
      } else if (vecStr.size() == 2) {
        sentenceID = Scan<long>(vecStr[0]);
        // phrase.CreateFromString(Output, staticData.GetOutputFactorOrder(), vecStr[1], staticData.GetFactorDelimiter(), NULL);
        phrase.CreateFromString(Output, staticData.GetOutputFactorOrder(), vecStr[1], NULL);
      } else {
        UTIL_THROW(util::Exception, "Reference file not loaded");
      }

      if (addBeginEndWord) {
        phrase.InitStartEndWord();
      }
      m_constraints[sentenceID].push_back(phrase);
    }
  }
}

std::vector<float> ConstrainedDecodingDeseg::DefaultWeights() const
{
  UTIL_THROW_IF2(m_numScoreComponents != 1,
                 "ConstrainedDecodingDeseg must only have 1 score");
  vector<float> ret(1, 1);
  return ret;
}

template <class H, class M>
const std::vector<Phrase> *GetConstraint(const std::map<long,std::vector<Phrase> > &constraints, const H &hypo)
{
  const M &mgr = hypo.GetManager();
  const InputType &input = mgr.GetSource();
  long id = input.GetTranslationId();

  map<long,std::vector<Phrase> >::const_iterator iter;
  iter = constraints.find(id);

  if (iter == constraints.end()) {
    UTIL_THROW(util::Exception, "Couldn't find reference " << id);

    return NULL;
  } else {
    return &iter->second;
  }
}

FFState* ConstrainedDecodingDeseg::EvaluateWhenApplied(
  const Hypothesis& hypo,
  const FFState* prev_state,
  ScoreComponentCollection* accumulator) const
{
  const std::vector<Phrase> *ref = GetConstraint<Hypothesis, Manager>(m_constraints, hypo);
  assert(ref);

  ConstrainedDecodingDesegState *ret = new ConstrainedDecodingDesegState(hypo);
  const Phrase &outputPhrase = ret->GetPhrase();

  size_t searchPos = NOT_FOUND;
  size_t i = 0;
  size_t size = 0;
  while(searchPos == NOT_FOUND && i < ref->size()) {
    searchPos = (*ref)[i].Find(outputPhrase, m_maxUnknowns);
    size = (*ref)[i].GetSize();
    i++;
  }

  float score;
  if (hypo.IsSourceCompleted()) {
    // translated entire sentence.
    bool match = (searchPos == 0) && (size == outputPhrase.GetSize());
    if (!m_negate) {
      score = match ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
    } else {
      score = !match ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
    }
  } else if (m_negate) {
    // keep all derivations
    score = 0;
  } else {
    score = (searchPos != NOT_FOUND) ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
  }

  accumulator->PlusEquals(this, score);

  return ret;
}

FFState* ConstrainedDecodingDeseg::EvaluateWhenApplied(
  const ChartHypothesis &hypo,
  int /* featureID - used to index the state in the previous hypotheses */,
  ScoreComponentCollection* accumulator) const
{
  const std::vector<Phrase> *ref = GetConstraint<ChartHypothesis, ChartManager>(m_constraints, hypo);
  assert(ref);

  const ChartManager &mgr = hypo.GetManager();
  const Sentence &source = static_cast<const Sentence&>(mgr.GetSource());

  ConstrainedDecodingDesegState *ret = new ConstrainedDecodingDesegState(hypo);
  const Phrase &outputPhrase = ret->GetPhrase();

  size_t searchPos = NOT_FOUND;
  size_t i = 0;
  size_t size = 0;
  while(searchPos == NOT_FOUND && i < ref->size()) {
    searchPos = (*ref)[i].Find(outputPhrase, m_maxUnknowns);
    size = (*ref)[i].GetSize();
    i++;
  }

  float score;
  if (hypo.GetCurrSourceRange().GetStartPos() == 0 &&
      hypo.GetCurrSourceRange().GetEndPos() == source.GetSize() - 1) {
    // translated entire sentence.
    bool match = (searchPos == 0) && (size == outputPhrase.GetSize());

    if (!m_negate) {
      score = match ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
    } else {
      score = !match ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
    }
  } else if (m_negate) {
    // keep all derivations
    score = 0;
  } else {
    score = (searchPos != NOT_FOUND) ? 0 : - ( m_soft ? 1 : std::numeric_limits<float>::infinity());
  }

  accumulator->PlusEquals(this, score);

  return ret;
}

void ConstrainedDecodingDeseg::SetParameter(const std::string& key, const std::string& value)
{
  if (key == "path") {
    m_paths = Tokenize(value, ",");
  } else if (key == "max-unknowns") {
    m_maxUnknowns = Scan<int>(value);
  } else if (key == "negate") {
    m_negate = Scan<bool>(value);
  } else if (key == "soft") {
    m_soft = Scan<bool>(value);
  } else {
    StatefulFeatureFunction::SetParameter(key, value);
  }
}

}

