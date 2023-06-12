#pragma once
#include <cmath>

using namespace std;
/**
@param
S = Q*k^T: [batch_size, head_num, seq_len_q, seq_len_k]
P (correlation): [batch_size, head_num, seq_len_q, seq_len_k]
padding_mask: [batch_size, seq_len_k], masked position is 1
**/
template <typename computeType, typename dataType>
void cpuAttentionMaskedSoftmax(dataType *S, dataType *P, const int* padding_mask, const int& batch_size, const int& seq_len_q, const int& seq_len_k, const int& head_num, const int& head_dim) {
  for (int i = 0; i < batch_size * seq_len_q * head_num; i++)
	{
		dataType* data_s = S + i * seq_len_k;
    dataType* data_p = P + i * seq_len_k;

		computeType _max = -INFINITY;
		computeType _sum = 0.0;
		computeType _scale = 1/sqrt(static_cast<computeType>(head_dim));
		
		for (int j = 0; j < seq_len_k; j++)
		{
			computeType s_val = (padding_mask[i / (seq_len_q * head_num) * seq_len_k + j % seq_len_k] == 1) ? -INFINITY : static_cast<computeType>(data_s[j]);
			_max = max(s_val * _scale, _max);
		}

		for (int j = 0; j < seq_len_k; j++)
		{
			_sum += (padding_mask[i / (seq_len_q * head_num) * seq_len_k + j % seq_len_k] == 1) ? static_cast<computeType>(0) : exp(static_cast<computeType>(data_s[j]) * _scale - _max);
		}

		for (int j = 0; j < seq_len_k; j++)
		{
			data_p[j] = (padding_mask[i / (seq_len_q * head_num) * seq_len_k + j % seq_len_k] == 1) ? static_cast<dataType>(0) : static_cast<dataType>(exp(static_cast<computeType>(data_s[j]) * _scale - _max) / _sum);
		}
		
	}
}
