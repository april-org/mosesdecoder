// $Id$

/***********************************************************************
Moses - factored phrase-based language decoder
Copyright (C) 2006 University of Edinburgh

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
***********************************************************************/

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include "lm/binary_format.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/left.hh"
#include "lm/model.hh"

#include "LanguageModelKen.h"
#include "LanguageModel.h"
#include "FFState.h"
#include "TypeDef.h"
#include "Util.h"
#include "FactorCollection.h"
#include "Phrase.h"
#include "InputFileStream.h"
#include "StaticData.h"
#include "ChartHypothesis.h"

#ifdef WITH_THREADS
#include <boost/scoped_ptr.hpp>
#endif

using namespace std;

namespace Moses {
namespace {

struct KenLMState : public FFState {
  lm::ngram::State state;
  int Compare(const FFState &o) const {
    const KenLMState &other = static_cast<const KenLMState &>(o);
    if (state.length < other.state.length) return -1;
    if (state.length > other.state.length) return 1;
    return std::memcmp(state.words, other.state.words, sizeof(lm::WordIndex) * state.length);
  }
};

/*
 * An implementation of single factor LM using Ken's code.
 */
template <class Model> class LanguageModelKen : public LanguageModel {
  public:
    LanguageModelKen(const std::string &file, ScoreIndexManager &manager, FactorType factorType, bool lazy);

    ~LanguageModelKen() {
#ifndef WITH_THREADS
      if (!--*m_refcount) {
        delete m_ngram;
        delete m_refcount;
      }
#endif
    }

    LanguageModel *Duplicate(ScoreIndexManager &scoreIndexManager) const;

    bool Useable(const Phrase &phrase) const {
      return (phrase.GetSize()>0 && phrase.GetFactor(0, m_factorType) != NULL);
    }

    std::string GetScoreProducerDescription(unsigned) const {
      std::ostringstream oss;
      oss << "LM_" << m_ngram->Order() << "gram";
      return oss.str();
    }

    const FFState *EmptyHypothesisState(const InputType &/*input*/) const {
      KenLMState *ret = new KenLMState();
      ret->state = m_ngram->BeginSentenceState();
      return ret;
    }

    void CalcScore(const Phrase &phrase, float &fullScore, float &ngramScore, size_t &oovCount) const;

    FFState *Evaluate(const Hypothesis &hypo, const FFState *ps, ScoreComponentCollection *out) const;

    FFState *EvaluateChart(const ChartHypothesis& cur_hypo, int featureID, ScoreComponentCollection *accumulator) const;

  private:
    LanguageModelKen(ScoreIndexManager &manager, const LanguageModelKen<Model> &copy_from);

    lm::WordIndex TranslateID(const Word &word) const {
      std::size_t factor = word.GetFactor(m_factorType)->GetId();
      return (factor >= m_lmIdLookup.size() ? 0 : m_lmIdLookup[factor]);
    }

    // Convert last words of hypothesis into vocab ids, returning an end pointer.  
    lm::WordIndex *LastIDs(const Hypothesis &hypo, lm::WordIndex *indices) const {
      lm::WordIndex *index = indices;
      lm::WordIndex *end = indices + m_ngram->Order() - 1;
      int position = hypo.GetCurrTargetWordsRange().GetEndPos();
      for (; ; ++index, --position) {
        if (position == -1) {
          *index = m_ngram->GetVocabulary().BeginSentence();
          return index + 1;
        }
        if (index == end) return index;
        *index = TranslateID(hypo.GetWord(position));
      }
    }

#ifdef WITH_THREADS
    boost::shared_ptr<Model> m_ngram;
#else
    Model *m_ngram;
    mutable unsigned int *m_refcount;
#endif
    std::vector<lm::WordIndex> m_lmIdLookup;

    FactorType m_factorType;

    const Factor *m_beginSentenceFactor;
};

class MappingBuilder : public lm::EnumerateVocab {
public:
  MappingBuilder(FactorCollection &factorCollection, std::vector<lm::WordIndex> &mapping)
    : m_factorCollection(factorCollection), m_mapping(mapping) {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    str_.assign(str.data(), str.size());
    std::size_t factorId = m_factorCollection.AddFactor(str_)->GetId();
    if (m_mapping.size() <= factorId) {
      // 0 is <unk> :-)
      m_mapping.resize(factorId + 1);
    }
    m_mapping[factorId] = index;
  }

private:
  FactorCollection &m_factorCollection;
  std::vector<lm::WordIndex> &m_mapping;

  std::string str_;
};

template <class Model> LanguageModelKen<Model>::LanguageModelKen(const std::string &file, ScoreIndexManager &manager, FactorType factorType, bool lazy) : m_factorType(factorType) {
  lm::ngram::Config config;
  IFVERBOSE(1) {
    config.messages = &std::cerr;
  } else {
    config.messages = NULL;
  }
  FactorCollection &collection = FactorCollection::Instance();
  MappingBuilder builder(collection, m_lmIdLookup);
  config.enumerate_vocab = &builder;
  config.load_method = lazy ? util::LAZY : util::POPULATE_OR_READ;

  try {
#ifdef WITH_THREADS
    m_ngram.reset(new Model(file.c_str(), config));
#else
    m_ngram = new Model(file.c_str(), config);
    m_refcount = new unsigned int();
    *m_refcount = 1;
#endif
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    abort();
  }

  m_beginSentenceFactor = collection.AddFactor(BOS_);
  Init(manager);
}

template <class Model> LanguageModel *LanguageModelKen<Model>::Duplicate(ScoreIndexManager &manager) const {
  return new LanguageModelKen<Model>(manager, *this);
}

template <class Model> LanguageModelKen<Model>::LanguageModelKen(ScoreIndexManager &manager, const LanguageModelKen<Model> &copy_from) :
    m_ngram(copy_from.m_ngram),
    // TODO: don't copy this.  
    m_lmIdLookup(copy_from.m_lmIdLookup),
    m_factorType(copy_from.m_factorType),
    m_beginSentenceFactor(copy_from.m_beginSentenceFactor) {
#ifndef WITH_THREADS
  m_refcount = copy_from.m_refcount;
  ++*m_refcount;
#endif
  Init(manager);
}

template <class Model> void LanguageModelKen<Model>::CalcScore(const Phrase &phrase, float &fullScore, float &ngramScore, size_t &oovCount) const {
  fullScore = 0;
  ngramScore = 0;
  oovCount = 0;

  if (!phrase.GetSize()) return;

  typename Model::State state_backing[2];
  typename Model::State *state0 = &state_backing[0], *state1 = &state_backing[1];
  size_t position;
  if (m_beginSentenceFactor == phrase.GetWord(0).GetFactor(m_factorType)) {
    *state0 = m_ngram->BeginSentenceState();
    position = 1;
  } else {
    *state0 = m_ngram->NullContextState();
    position = 0;
  }
  
  size_t ngramBoundary = m_ngram->Order() - 1;

  for (; position < phrase.GetSize(); ++position) {
    const Word &word = phrase.GetWord(position);
    if (word.IsNonTerminal()) {
      *state0 = m_ngram->NullContextState();
    } else {
      lm::WordIndex index = TranslateID(word);
      float score = TransformLMScore(m_ngram->Score(*state0, index, *state1));
      std::swap(state0, state1);
      if (position >= ngramBoundary) ngramScore += score;
      fullScore += score;
      if (!index) ++oovCount;
    }
  }
}

template <class Model> FFState *LanguageModelKen<Model>::Evaluate(const Hypothesis &hypo, const FFState *ps, ScoreComponentCollection *out) const {
  const lm::ngram::State &in_state = static_cast<const KenLMState&>(*ps).state;

  std::auto_ptr<KenLMState> ret(new KenLMState());
  
  if (!hypo.GetCurrTargetLength()) {
    ret->state = in_state;
    return ret.release();
  }

  const std::size_t begin = hypo.GetCurrTargetWordsRange().GetStartPos();
  //[begin, end) in STL-like fashion.
  const std::size_t end = hypo.GetCurrTargetWordsRange().GetEndPos() + 1;
  const std::size_t adjust_end = std::min(end, begin + m_ngram->Order() - 1);

  std::size_t position = begin;
  typename Model::State aux_state;
  typename Model::State *state0 = &ret->state, *state1 = &aux_state;

  float score = m_ngram->Score(in_state, TranslateID(hypo.GetWord(position)), *state0);
  ++position;
  for (; position < adjust_end; ++position) {
    score += m_ngram->Score(*state0, TranslateID(hypo.GetWord(position)), *state1);
    std::swap(state0, state1);
  }

  if (hypo.IsSourceCompleted()) {
    // Score end of sentence.  
    std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
    const lm::WordIndex *last = LastIDs(hypo, &indices.front());
    score += m_ngram->FullScoreForgotState(&indices.front(), last, m_ngram->GetVocabulary().EndSentence(), ret->state).prob;
  } else if (adjust_end < end) {
    // Get state after adding a long phrase.  
    std::vector<lm::WordIndex> indices(m_ngram->Order() - 1);
    const lm::WordIndex *last = LastIDs(hypo, &indices.front());
    m_ngram->GetState(&indices.front(), last, ret->state);
  } else if (state0 != &ret->state) {
    // Short enough phrase that we can just reuse the state.  
    ret->state = *state0;
  }

  score = TransformLMScore(score);

  if (OOVFeatureEnabled()) {
    std::vector<float> scores(2);
    scores[0] = score;
    scores[1] = 0.0;
    out->PlusEquals(this, scores);
  } else {
    out->PlusEquals(this, score);
  }

  return ret.release();
}

class LanguageModelChartStateKenLM : public FFState {
  public:
    LanguageModelChartStateKenLM() {}

    const lm::ngram::ChartState &GetChartState() const { return m_state; }
    lm::ngram::ChartState &GetChartState() { return m_state; }

    int Compare(const FFState& o) const
    {
      const LanguageModelChartStateKenLM &other = static_cast<const LanguageModelChartStateKenLM&>(o);
      int ret = m_state.Compare(other.m_state);
      return ret;
    }

  private:
    lm::ngram::ChartState m_state;
};

template <class Model> FFState *LanguageModelKen<Model>::EvaluateChart(const ChartHypothesis& hypo, int featureID, ScoreComponentCollection *accumulator) const {
  LanguageModelChartStateKenLM *newState = new LanguageModelChartStateKenLM();
  lm::ngram::RuleScore<Model> ruleScore(*m_ngram, newState->GetChartState());
  const AlignmentInfo::NonTermIndexMap &nonTermIndexMap = hypo.GetCurrTargetPhrase().GetAlignmentInfo().GetNonTermIndexMap();

  const size_t size = hypo.GetCurrTargetPhrase().GetSize();
  size_t phrasePos = 0;
  // Special cases for first word.  
  if (size) {
    const Word &word = hypo.GetCurrTargetPhrase().GetWord(0);
    if (word.GetFactor(m_factorType) == m_beginSentenceFactor) {
      // Begin of sentence
      ruleScore.BeginSentence();
      phrasePos++;
    } else if (word.IsNonTerminal()) {
      // Non-terminal is first so we can copy instead of rescoring.  
      const ChartHypothesis *prevHypo = hypo.GetPrevHypo(nonTermIndexMap[phrasePos]);
      const lm::ngram::ChartState &prevState = static_cast<const LanguageModelChartStateKenLM*>(prevHypo->GetFFState(featureID))->GetChartState();
      ruleScore.BeginNonTerminal(prevState, prevHypo->GetScoreBreakdown().GetScoresForProducer(this)[0]);
      phrasePos++;
    }
  }

  for (; phrasePos < size; phrasePos++) {
    const Word &word = hypo.GetCurrTargetPhrase().GetWord(phrasePos);
    if (word.IsNonTerminal()) {
      const ChartHypothesis *prevHypo = hypo.GetPrevHypo(nonTermIndexMap[phrasePos]);
      const lm::ngram::ChartState &prevState = static_cast<const LanguageModelChartStateKenLM*>(prevHypo->GetFFState(featureID))->GetChartState();
      ruleScore.NonTerminal(prevState, prevHypo->GetScoreBreakdown().GetScoresForProducer(this)[0]);
    } else {
      ruleScore.Terminal(TranslateID(word));
    }
  }

  accumulator->Assign(this, ruleScore.Finish());
  return newState;
}

} // namespace

LanguageModel *ConstructKenLM(const std::string &file, ScoreIndexManager &manager, FactorType factorType, bool lazy) {
  lm::ngram::ModelType model_type;
  if (lm::ngram::RecognizeBinary(file.c_str(), model_type)) {
    switch(model_type) {
    case lm::ngram::HASH_PROBING:
      return new LanguageModelKen<lm::ngram::ProbingModel>(file, manager, factorType, lazy);
    case lm::ngram::TRIE_SORTED:
      return new LanguageModelKen<lm::ngram::TrieModel>(file, manager, factorType, lazy);
    case lm::ngram::QUANT_TRIE_SORTED:
      return new LanguageModelKen<lm::ngram::QuantTrieModel>(file, manager, factorType, lazy);
    case lm::ngram::ARRAY_TRIE_SORTED:
      return new LanguageModelKen<lm::ngram::ArrayTrieModel>(file, manager, factorType, lazy);
    case lm::ngram::QUANT_ARRAY_TRIE_SORTED:
      return new LanguageModelKen<lm::ngram::QuantArrayTrieModel>(file, manager, factorType, lazy);
    default:
      std::cerr << "Unrecognized kenlm model type " << model_type << std::endl;
      abort();
    }
  } else {
    return new LanguageModelKen<lm::ngram::ProbingModel>(file, manager, factorType, lazy);
  }
}

}

