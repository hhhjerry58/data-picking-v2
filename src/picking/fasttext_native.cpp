// Thin resumable adapter around the pinned upstream fastText supervised SGD.
// Update, loss, n-gram hashing and dictionary remain upstream. See VENDOR_PATCHES.md
// for the complete-initialization bug fix and training-state serialization.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <cmath>
#include "fasttext.h"
#include "loss.h"

namespace py = pybind11;

template <typename T> void writeValue(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template <typename T> T readValue(std::istream& in) {
  T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("Truncated fastText trainer state");
  return value;
}

class ResumableFastText : public fasttext::FastText {
 private:
  std::unique_ptr<fasttext::Model::State> state;
  std::ifstream stream;
  int64_t localTokens = 0;
  int64_t updates = 0;

 public:
  void initialize(const std::vector<std::string>& argv, const std::string& modelPath = "",
                  const std::string& statePath = "") {
    fasttext::Args requested;
    requested.parseArgs(argv);
    if (requested.thread != 1 || requested.model != fasttext::model_name::sup ||
        requested.loss != fasttext::loss_name::softmax || !requested.pretrainedVectors.empty())
      throw std::invalid_argument("Adapter requires single-thread supervised softmax without pretrained vectors");
    if (!modelPath.empty()) {
      loadModel(modelPath);
      if (requested.dim != args_->dim || requested.wordNgrams != args_->wordNgrams || requested.bucket != args_->bucket)
        throw std::runtime_error("Restored model arguments do not match");
      *args_ = requested; // Dictionary shares this Args object.
    } else {
      args_ = std::make_shared<fasttext::Args>(requested);
      dict_ = std::make_shared<fasttext::Dictionary>(args_);
      std::ifstream training(args_->input);
      if (!training) throw std::runtime_error("Cannot open fastText training data");
      dict_->readFromFile(training);
      input_ = createRandomMatrix();
      output_ = createTrainOutputMatrix();
      quant_ = false;
      auto loss = createLoss(output_);
      model_ = std::make_shared<fasttext::Model>(input_, output_, loss, true);
    }
    if (dict_->ntokens() <= 0 || dict_->nlabels() != 2)
      throw std::runtime_error("Expected a nonempty two-label dataset");
    state = std::make_unique<fasttext::Model::State>(args_->dim, output_->size(0), args_->seed);
    stream.open(args_->input, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot reopen fastText training data");
    tokenCount_ = 0; localTokens = 0; updates = 0; trainException_ = nullptr;
    if (!statePath.empty()) {
      std::ifstream in(statePath, std::ios::binary);
      const auto magic = readValue<uint64_t>(in);
      if (magic != 0x4654524553554D31ULL) throw std::runtime_error("Unknown trainer state version");
      if (readValue<int64_t>(in) != dict_->ntokens() || readValue<int>(in) != args_->epoch ||
          readValue<int>(in) != args_->seed || readValue<double>(in) != args_->lr ||
          readValue<int>(in) != args_->lrUpdateRate)
        throw std::runtime_error("Optimizer/schedule configuration changed during resume");
      tokenCount_ = readValue<int64_t>(in);
      localTokens = readValue<int64_t>(in);
      updates = readValue<int64_t>(in);
      int64_t position = readValue<int64_t>(in);
      int flags = readValue<int>(in);
      state->loadTrainingState(in);
      stream.seekg(position);
      stream.setstate(static_cast<std::ios::iostate>(flags));
    }
  }

  void advance(int untilEpoch, int64_t maxAdditionalUpdates = 0) {
    if (!state || !stream.is_open()) throw std::runtime_error("Trainer was not initialized");
    const int64_t limit = std::min(untilEpoch, args_->epoch) * dict_->ntokens();
    const int64_t initialUpdates = updates;
    std::vector<int32_t> line, labels;
    while (tokenCount_ < limit) {
      fasttext::real progress = fasttext::real(tokenCount_) / (args_->epoch * dict_->ntokens());
      fasttext::real lr = args_->lr * (1.0 - progress);
      localTokens += dict_->getLine(stream, line, labels);
      supervised(*state, lr, line, labels);
      ++updates;
      if (localTokens > args_->lrUpdateRate) {
        tokenCount_ += localTokens;
        localTokens = 0;
      }
      if (!std::isfinite(state->getLoss())) throw std::runtime_error("Non-finite fastText training loss");
      if (maxAdditionalUpdates > 0 && updates - initialUpdates >= maxAdditionalUpdates) break;
    }
  }

  void save(const std::string& modelPath, const std::string& statePath) {
    saveModel(modelPath);
    auto flags = stream.rdstate();
    stream.clear();
    auto position = stream.tellg();
    if (position < 0) throw std::runtime_error("Cannot determine fastText input cursor");
    stream.setstate(flags);
    std::ofstream out(statePath, std::ios::binary);
    writeValue<uint64_t>(out, 0x4654524553554D31ULL);
    writeValue<int64_t>(out, dict_->ntokens());
    writeValue<int>(out, args_->epoch);
    writeValue<int>(out, args_->seed);
    writeValue<double>(out, args_->lr);
    writeValue<int>(out, args_->lrUpdateRate);
    writeValue<int64_t>(out, tokenCount_);
    writeValue<int64_t>(out, localTokens);
    writeValue<int64_t>(out, updates);
    writeValue<int64_t>(out, static_cast<int64_t>(position));
    writeValue<int>(out, static_cast<int>(flags));
    state->saveTrainingState(out);
    out.close();
    if (!out) throw std::runtime_error("Could not save complete fastText trainer state");
  }

  std::map<std::string, double> status() const {
    return {{"committed_tokens", static_cast<double>(tokenCount_)},
            {"pending_tokens", static_cast<double>(localTokens)},
            {"updates", static_cast<double>(updates)},
            {"tokens_per_epoch", static_cast<double>(dict_->ntokens())},
            {"epoch_progress", static_cast<double>(tokenCount_) / dict_->ntokens()},
            {"loss", updates ? state->getLoss() : 0.0},
            {"finished", tokenCount_ >= args_->epoch * dict_->ntokens() ? 1.0 : 0.0}};
  }

  std::map<std::string, double> inputStats() const {
    const auto matrix = getInputMatrix();
    int64_t zeros = 0, nonfinite = 0;
    double absoluteMax = 0;
    for (int64_t row = 0; row < matrix->rows(); ++row) {
      for (int64_t column = 0; column < matrix->cols(); ++column) {
        const auto value = matrix->at(row, column);
        zeros += value == 0;
        nonfinite += !std::isfinite(value);
        absoluteMax = std::max(absoluteMax, std::abs(static_cast<double>(value)));
      }
    }
    return {{"elements", static_cast<double>(matrix->rows()*matrix->cols())},
            {"zeros", static_cast<double>(zeros)}, {"nonfinite", static_cast<double>(nonfinite)},
            {"absolute_max", absoluteMax}};
  }

  std::vector<double> probabilities(const std::vector<std::string>& texts) const {
    std::vector<double> output;
    output.reserve(texts.size());
    for (const auto& text : texts) {
      std::istringstream input(text + "\n");
      std::vector<std::pair<fasttext::real, std::string>> predictions;
      predictLine(input, predictions, -1, 0.0);
      bool found = false;
      for (const auto& prediction : predictions) {
        if (prediction.second == "__label__include") {
          output.push_back(prediction.first);
          found = true; break;
        }
      }
      if (!found) throw std::runtime_error("Include probability missing");
    }
    return output;
  }
};

PYBIND11_MODULE(_fasttext_native, m) {
  py::class_<ResumableFastText>(m, "Classifier")
      .def(py::init<>())
      .def("initialize", &ResumableFastText::initialize, py::arg("argv"), py::arg("model_path") = "", py::arg("state_path") = "")
      .def("advance", &ResumableFastText::advance, py::arg("until_epoch"), py::arg("max_additional_updates") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def("save", &ResumableFastText::save)
      .def("status", &ResumableFastText::status)
      .def("input_stats", &ResumableFastText::inputStats)
      .def("load_model", py::overload_cast<const std::string&>(&ResumableFastText::loadModel))
      .def("predict_include", &ResumableFastText::probabilities, py::call_guard<py::gil_scoped_release>());
}
