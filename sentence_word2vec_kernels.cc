/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include <csignal>


namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 1000;

namespace {

bool ScanWord(StringPiece* input, string* word) {
  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;
  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace

class SkipgramSentenceOp : public OpKernel {
 public:
  explicit SkipgramSentenceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

    mutex_lock l(mu_);
    example_pos_ = sentence_size_;
    corpus_sentences_index_ = corpus_sentences_size_;
    label_pos_ = sentence_size_;
    for (int i = 0; i < kPrecalc; ++i) {
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            NextExample(&precalc_examples_[j].input,
                &precalc_examples_[j].label);
          }
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }
    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  int min_count_ = 5;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor freq_;
  int32 corpus_size_ = 0;
  int32 corpus_sentences_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<std::vector<int32>> corpus_sentences_;
  std::vector<Example> precalc_examples_;
  int precalc_index_ = 0;
  std::vector<int32> sentence_;
  std::vector<std::vector<int32>> sentences_;
  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int64 total_sentences_processed_ GUARDED_BY(mu_) = 0;
  int32 example_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  // index for which sentence we're at in the corpus
  int32 corpus_sentences_index_ GUARDED_BY(mu_) = 0;
  // size of the current sentence
  int32 sentence_size_ GUARDED_BY(mu_) =
    corpus_sentences_[corpus_sentences_index_].size();
  // number of times we have processed an example
  int32 example_counter_ GUARDED_BY(mu_) = 0;

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ is reset at the end of the sentence. For each
  // example, we randomly generate label_pos_ from [0, sentence_size_) for
  // labels. I.e., the context window is the rest of the words in a sentence.
  void NextExample(int32* example, int32* label) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (example_counter_ >= sentence_size_ - 1) {
        ++example_pos_;
        ++total_words_processed_;
        example_counter_ = 0;
        if (example_pos_ >= sentence_size_) {
          example_pos_ = 0;
          ++total_sentences_processed_;
          ++corpus_sentences_index_;
          if (corpus_sentences_index_ >= corpus_sentences_size_) {
            ++current_epoch_;
            corpus_sentences_index_ = 0;
          }
          sentence_size_ = corpus_sentences_[corpus_sentences_index_].size();
        }
      }
      ++example_counter_;
      if (subsample_ > 0) {
        int32 word_freq =
          freq_.flat<int32>()(corpus_sentences_[corpus_sentences_index_][example_pos_]);
        // See Eq. 5 in http://arxiv.org/abs/1310.4546
        float keep_prob =
          (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
          (subsample_ * corpus_size_) / word_freq;
        if (rng_.RandFloat() > keep_prob) {
          break;
        }
      }
      while (true) {
        label_pos_ = rng_.Uniform(sentence_size_);
        if (example_pos_ != label_pos_) {
          break;
        }
      }
      break;
    }
    *example = corpus_sentences_[corpus_sentences_index_][example_pos_];
    *label = corpus_sentences_[corpus_sentences_index_][label_pos_];
  }

  Status Init(Env* env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    // the input file is assumed to be a newline-separated list of sentences
    std::vector<string> sentences_str = str_util::Split(data, '\n');
    sentences_str.erase(sentences_str.begin() + sentences_str.size() - 1);
    std::vector<StringPiece> sentences;
    string w;
    corpus_size_ = 0;
    // need to convert vector<string> to vector<StringPiece>
    for (std::size_t i = 0; i < sentences_str.size(); ++i) {
      sentences.push_back(sentences_str[i]);
    }
    std::vector<StringPiece> input = sentences;
    std::unordered_map<string, int32> word_freq;
    for (std::size_t i = 0; i < input.size(); ++i) {
      while (ScanWord(&input[i], &w)) {
        ++(word_freq[w]);
        ++corpus_size_;
      }
    }
    corpus_sentences_size_ = sentences.size();
    if (corpus_sentences_size_ < 10) {
      return errors::InvalidArgument("The text file ", filename,
          " contains too little data: ",
          corpus_sentences_size_, " sentences");
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }
    LOG(INFO) << "Data file: " << filename << " contains " << data.size()
      << " bytes, " << corpus_sentences_size_ << " sentences, "
      << corpus_size_ << " words, " << word_freq.size()
      << " unique words, " << ordered.size()
      << " unique frequent words.";
    word_freq.clear();
    std::sort(ordered.begin(), ordered.end(),
        [](const WordFreq& x, const WordFreq& y) {
        return x.second > y.second;
        });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK";
    static const int32 kUnkId = 0;
    std::unordered_map<string, int32> word_id;
    int64 total_counted = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      auto id = i + 1;
      word.flat<string>()(id) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
    }
    freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
    word_ = word;
    freq_ = freq;
    corpus_.reserve(corpus_size_);
    std::vector<int32> tmp;
    input = sentences;
    for (std::size_t i = 0; i < input.size(); ++i) {
      int j = 0;
      while (ScanWord(&input[i], &w)) {
        if (j == 0) {
          corpus_sentences_.push_back(tmp);
        }
        corpus_sentences_[i].push_back(gtl::FindWithDefault(word_id, w, kUnkId));
        j++;
      }
    }
    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("SkipgramSentence").Device(DEVICE_CPU), SkipgramSentenceOp);

}  // end namespace tensorflow
