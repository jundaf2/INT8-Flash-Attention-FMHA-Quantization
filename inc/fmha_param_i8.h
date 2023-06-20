#pragma once

struct AttnDataDescriptor{
  int batch_num = 1;
  int seq_len = 64;
  int head_num = 4;
  int head_dim = 64;
};

struct FMHAParamI8 {
  //
  float q_amax = 0.0f;
  float k_amax = 0.0f;
  float v_amax = 0.0f;
  float o_amax = 1.0f;
  float s_max = 1.0f;
};
