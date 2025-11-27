#include <Arduino.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model.h"

// Constants
static const int T = 40;
static const int N_FEATURES = 4;
static const int INPUT_LEN = T * N_FEATURES;

// Buffers 
constexpr int kTensorArenaSize = 64 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

static const size_t kCsvBufSize = 4096;
static char csv_buf[kCsvBufSize];

// TFLM variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Pretty-print 
const char* TfTypeName(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "FLOAT32";
    case kTfLiteInt8:    return "INT8";
    case kTfLiteUInt8:   return "UINT8";
    case kTfLiteInt16:   return "INT16";
    case kTfLiteInt32:   return "INT32";
    default:             return "OTHER";
  }
}

void printDims(const TfLiteTensor* t) {
  Serial.print("[");
  for (int i = 0; i < t->dims->size; ++i) {
    Serial.print(t->dims->data[i]);
    if (i + 1 < t->dims->size) Serial.print(",");
  }
  Serial.print("]");
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  delay(100);

  Serial.println(F("Nano HuGaDB Inference Ready"));

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("Schema mismatch: model="));
    Serial.print(model->version());
    Serial.print(F(" tflite_micro="));
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1) { delay(1000); }
  }

  static tflite::AllOpsResolver resolver;  
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println(F("AllocateTensors() failed — try increasing tensor arena"));
    while (1) { delay(1000); }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print(F("Input type="));  Serial.println((int)input->type);
  Serial.print(F("Input dims="));  printDims(input);  Serial.println();
  Serial.print(F("Output type=")); Serial.println((int)output->type);
  Serial.print(F("Output dims=")); printDims(output); Serial.println();

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    Serial.print(F("In zp=")); Serial.print(input->params.zero_point);
    Serial.print(F(" scale=")); Serial.println(input->params.scale, 7);
  }
  if (output->type == kTfLiteInt8 || output->type == kTfLiteUInt8) {
    Serial.print(F("Out zp=")); Serial.print(output->params.zero_point);
    Serial.print(F(" scale=")); Serial.println(output->params.scale, 7);
  }

  Serial.println(F("Awaiting CSV line (160 floats)…"));
}

bool readCsvLine(char* buf, size_t cap, size_t& out_len) {
  out_len = 0;
  while (true) {
    if (!Serial.available()) {
      delay(1);
      continue;
    }
    int c = Serial.read();
    if (c < 0) continue;
    if (c == '\r') continue;
    if (c == '\n') {
      buf[out_len] = '\0';
      return true;
    }
    if (out_len + 1 >= cap) {
      // Overflow ... Consume till EOL
      out_len = 0;
      while (Serial.available()) {
        int d = Serial.read();
        if (d == '\n') break;
      }
      Serial.println(F("Line too long; dropped. Send exactly 160 floats per line."));
      Serial.println(F("Awaiting CSV line (160 floats)…"));
      return false;
    }
    buf[out_len++] = (char)c;
  }
}

int splitCsvToFloats(const char* line, float* out, int maxN) {
  int n = 0;
  const char* p = line;
  while (*p && n < maxN) {
    while (*p == ' ' || *p == '\t') ++p;
    char* endp = nullptr;
    float v = strtof(p, &endp);
    if (endp == p) {
      if (*p == ',') { ++p; continue; }
      break;
    }
    out[n++] = v;
    p = endp;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') ++p;
  }
  return n;
}

template<typename T>
T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

void loop() {
  size_t nlen = 0;
  if (!readCsvLine(csv_buf, kCsvBufSize, nlen)) {
    return;
  }
  if (nlen == 0) return;

  // Parse floats
  static float feats[INPUT_LEN];
  int n = splitCsvToFloats(csv_buf, feats, INPUT_LEN);
  if (n != INPUT_LEN) {
    Serial.print(F("Bad CSV count: got "));
    Serial.print(n);
    Serial.print(F(", need "));
    Serial.println(INPUT_LEN);
    Serial.println(F("Awaiting CSV line (160 floats)…"));
    return;
  }

  // Load input tensor
  if (input->type == kTfLiteInt8) {
    const float s = input->params.scale;
    const int zp = input->params.zero_point;
    int8_t* dst = input->data.int8;
    for (int i = 0; i < INPUT_LEN; ++i) {
      const float q = feats[i] / s + zp;
      const int qi = (int)lrintf(q);
      dst[i] = (int8_t)clamp(qi, -128, 127);
    }
  } else if (input->type == kTfLiteFloat32) {
    float* dst = input->data.f;
    for (int i = 0; i < INPUT_LEN; ++i) dst[i] = feats[i];
  } else {
    Serial.print(F("Unsupported input type: "));
    Serial.println(TfTypeName(input->type));
    return;
  }

  // Invoke
  TfLiteStatus st = interpreter->Invoke();
  if (st != kTfLiteOk) {
    Serial.println(F("Invoke failed."));
    return;
  }

  int num_classes = (output->dims->size >= 2) ? output->dims->data[1] : 1;

  int pred = -1;
  if (output->type == kTfLiteInt8) {
    int8_t* o = output->data.int8;
    int8_t mv = o[0];
    pred = 0;
    for (int i = 1; i < num_classes; ++i) {
      if (o[i] > mv) { mv = o[i]; pred = i; }
    }
  } else if (output->type == kTfLiteFloat32) {
    float* o = output->data.f;
    float mv = o[0];
    pred = 0;
    for (int i = 1; i < num_classes; ++i) {
      if (o[i] > mv) { mv = o[i]; pred = i; }
    }
  } else {
    Serial.print(F("Unsupported output type: "));
    Serial.println(TfTypeName(output->type));
    return;
  }

  Serial.print(F("Pred: "));
  Serial.println(pred);
}