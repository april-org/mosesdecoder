#pragma once

#include "SingleFactor.h"
#include <unordered_map>
#include <../../opt/eigen/Eigen/Dense>
#include <fstream>

using namespace Eigen;
using namespace std;

namespace Moses {
  
  class SimpleNeuralLM : public LanguageModelSingleFactor {
  protected:

    // they are loaded but later replaced by the premultiplied input matrix
    Matrix<float,Dynamic,Dynamic,Eigen::RowMajor> input_word_embeddings;

    Matrix<float,Dynamic,Dynamic> hidden_layer_weights;
    Matrix<float,Dynamic,1> hidden_layer_biases;

    Matrix<float,Dynamic,Dynamic> output_layer_weights;
    Matrix<float,Dynamic,1> output_layer_biases;

    Matrix<float,Dynamic,Dynamic> auxilliary_layer_weights;
    Matrix<float,Dynamic,1> auxiliary_layer_biases;
    
    typedef unordered_map<std::string, int> WordId;
    WordId input_word_index;
    int m_unk; // index of unknown word

    int ngram_size;
    int input_vocab_size;
    int output_vocab_size;
    int input_embedding_dimension;
    int num_hidden;
    enum activation_function_type { Tanh, HardTanh, Rectifier, Identity, InvalidFunction };
    activation_function_type activation;

    int lookup_word(const std::string &word, int unkid) const;
    float lookup_ngram(const vector<int> &words) const;

    void load_model(const string &filename);
    void readWordsFile(ifstream &file, vector<string> &word_list);
    void readMatrix(ifstream &file, Eigen::MatrixBase<Derived> &param_const);
    void readConfig(ifstream &file);



  public:
    SimpleNeuralLM(const std::string &line);
    ~SimpleNeuralLM();
    virtual LMResult GetValue(const std::vector<const Word*> &contextFactor, State* finalState = 0) const;
    virtual void Load(AllOptions::ptr const& opts);
  };
  
  
} // namespace

