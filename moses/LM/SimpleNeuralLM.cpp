
#include "moses/StaticData.h"
#include "moses/FactorCollection.h"
#include <boost/functional/hash.hpp>
#include "SimpleNeuralLM.h"

#include <sstream>


namespace Moses {

  SimpleNeuralLM::SimpleNeuralLM(const string &line)
    :LanguageModelSingleFactor(line) {
    ReadParameters();
  }


  SimpleNeuralLM::~SimpleNeuralLM() {
    // TODO
  }


  void SimpleNeuralLM::Load(AllOptions::ptr const& opts) {


    // Set parameters required by ancestor classes
    FactorCollection &factorCollection = FactorCollection::Instance();
    m_sentenceStart = factorCollection.AddFactor(Output, m_factorType, BOS_);
    m_sentenceStartWord[m_factorType] = m_sentenceStart;
    m_sentenceEnd		= factorCollection.AddFactor(Output, m_factorType, EOS_);
    m_sentenceEndWord[m_factorType] = m_sentenceEnd;
    
    load_model(m_filePath); // m_filePath is an inherited attribute
    m_unk = lookup_word("<unk>");

    // TODO review this part
    UTIL_THROW_IF2(m_nGramOrder != m_neuralLM_shared->get_order(),
		   "Wrong order of neuralLM: LM has " << m_neuralLM_shared->get_order() 
		   << ", but Moses expects " << m_nGramOrder);
  
  }

  void SimpleNeuralLM::readWordsFile(ifstream &file, vector<string> &word_list) {
    string line;
    while (getline(file, line) && line != "") {
      stringstream ss(line);
      // we could check that there is only one word per line
      string word;
      ss >> word;
      word_list.push_back(word);
    }
  }

  void SimpleNeuralLM::readMatrix(ifstream &file,
				  Eigen::MatrixBase<Derived> &param_const) {
    Eigen::MatrixBase<Derived> &param = const_cast<Eigen::MatrixBase<Derived>&>(c);
    int i = 0;
    int j = 0;
    //const int rows = param.rows();
    const int cols = param.cols();
    string line;
    while (getline(file, line) && line != "") {
      stringstream ss(line);
      float weight;
      while (ss >> weight) {
	param(i,j) = weight;
	++j;
	if (j>=cols) {
	  j=0;
	  ++i;
	}
      }
    }
  }

  void SimpleNeuralLM::readConfig(ifstream &file) {
    string line;
    while (getline(file, line) && line != "") {
      stringstream ss(line);
      string token;
      ss >> token;
      if (token == "ngram_size")
	ss >> ngram_size;
      else if (token == "vocab_size") {
	ss >> input_vocab_size;
	output_vocab_size = input_vocab_size;
      } else if (token == "input_vocab_size")
	ss >> input_vocab_size;
      else if (token == "output_vocab_size")
	ss >> output_vocab_size;
      else if (token == "input_embedding_dimension")
	ss >> input_embedding_dimension;
      else if (token == "num_hidden")
	ss >> num_hidden;
      else if (token == "output_embedding_dimension")
	ss >> output_embedding_dimension;
      else if (token == "activation_function") {
	string activation;
	ss >> activation;
	if (s == "identity")
	  activation = Identity;
	else if (s == "rectifier")
	  activation = Rectifier;
	else if (s == "tanh")
	  activation = Tanh;
	else if (s == "hardtanh")
	  activation = HardTanh;
	else
	  activation = InvalidFunction;
      } else if (token == "version") {
	int version;
	ss >> version;
	if (version != 1) {
	  cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
	  exit(1);
	}
      } else if (token == "auxiliary_softmax") {
	ss >> num_hidden;
      } else
	cerr << "warning: unrecognized field in config: " << token << endl;
    }

    // use these values to resize matrices:

    //Resizes to the given size, and sets all coefficients in this expression to zero.
    

    
  } // closes method

  void SimpleNeuralLM::load_model(const string &filename) {    
    ifstream file(filename.c_str());
    if (!file) throw runtime_error("Could not open file " + filename);

    Matrix<float,Dynamic,Dynamic,Eigen::RowMajor> input_word_embeddings;

    string line;
    while (getline(file, line)) {
      if (line == "\\config") {
	readConfig(file);
	// use config values to resize some matrices:

        // resize(ngram_size,
        //     input_vocab_size,
        //     output_vocab_size,
        //     input_embedding_dimension,
        //     num_hidden,
        //     output_embedding_dimension);


	//Resizes to the given size, and sets all coefficients in this expression to zero.
	input_layer.setZero(input_vocab_size, input_embedding_dimension, ngram_size-1);

    first_hidden_linear.resize(num_hidden, input_embedding_dimension*(ngram_size-1));
    first_hidden_activation.resize(num_hidden);



      } else if (line == "\\vocab") {
	input_words.clear();
	readWordsFile(file, input_words);
	output_words = input_words;
      } else if (line == "\\input_vocab")	{
	input_words.clear();
	readWordsFile(file, input_words);
      } else if (line == "\\output_vocab") {
	output_words.clear();
	readWordsFile(file, output_words);
      } else if (line == "\\input_embeddings") {
	input_layer.read(file);
      } else if (line == "\\hidden_weights 1") {
	readMatrix(file,first_hidden_weights);
      } else if (line == "\\hidden_biases 1") {
	readMatrix(file,first_hidden_biases);
      } else if (line == "\\hidden_weights 2") {
	readMatrix(file,second_hidden_weights);
      } else if (line == "\\hidden_biases 2") {
	readMatrix(file,second_hidden_biases);
      } else if (line == "\\output_weights") {
	readMatrix(file,output_weights);
      } else if (line == "\\output_biases") {
	readMatrix(file,output_biases);
      }	else if (line == "\\auxiliary_softmax_vocab") {
	// TODO
      } else if (line == "\\auxiliary_hidden_biases 1") {
	readMatrix(file,auxiliary_hidden_biases);
      } else if (line == "\\auxiliary_hidden_weights 1") {
	readMatrix(file,auxiliary_hidden_weights);
      } else if (line == "\\auxiliary_output_weights") {
	auxiliary_output_layer.read_weights(file);
      } else if (line == "\\auxiliary_output_biases") {
	auxiliary_output_layer.read_biases(file);
      } else if (line == "\\end") {
	break;
      } else if (line == "") {
	continue;
      } else {
	cerr << "warning: unrecognized section: " << line << endl;
	// skip over section
	while (getline(file, line) && line != "") { }
      }
    } // closes while (getline(file, line))
    file.close();
    
    // COMPLETE


    // Since input and first_hidden_linear are both linear,
  // we can multiply them into a single linear layer *if* we are not training
  int context_size = ngram_size-1;
  Matrix<user_data_t,Dynamic,Dynamic> U = first_hidden_linear.U;
  if (num_hidden == 0)
  {
    first_hidden_linear.U.resize(output_embedding_dimension, input_vocab_size * context_size);
  }
  else
  {
    first_hidden_linear.U.resize(num_hidden, input_vocab_size * context_size);
  }
  for (int i=0; i<context_size; i++)
    first_hidden_linear.U.middleCols(i*input_vocab_size, input_vocab_size) = U.middleCols(i*input_embedding_dimension, input_embedding_dimension) * input_layer.W->transpose();
  input_layer.W->resize(1,1); // try to save some memory
  premultiplied = true;


    
  } // closes method

  int SimpleNeuralLM::lookup_word(const string &word, int unkid) const {
    auto pos = input_word_index.find(word);
    return pos == input_word_index.end() ? unkid : pos->second;
  }

  float SimpleNeuralLM::lookup_ngram(const vector<int> &words) const {
    
  }
  
  LMResult SimpleNeuralLM::GetValue(const vector<const Word*> &contextFactor,
				    State* finalState) const {
    
    vector<int> words(contextFactor.size());
    const size_t n=contextFactor.size();
    for (size_t i=0; i<n; i++) {
      const Word* word = contextFactor[i];
      const Factor* factor = word->GetFactor(m_factorType);
      const string string = factor->GetString().as_string();
      int neuralLM_wordID = lookup_word(string);
      words[i] = neuralLM_wordID;
    }
    
    // CAUTION: this approach has problems since different ngram
    // contexts may lead to the same hash value. It is possible to
    // implement a solution without this problem by means (for
    // instance) of a trie
    size_t hashCode = 0; 
    for (size_t i=1; i<n; i++) { // OBSERVE THAT WE START AT 1
      boost::hash_combine(hashCode, words[i]);
    }

    float value = lookup_ngram(words);
    
    // Create a new struct to hold the result
    LMResult ret;
    ret.score = FloorScore(value);
    ret.unknown = (words.back() == m_unk);
    
    (*finalState) = (State*) hashCode;
    
    return ret;
  }
  
}


