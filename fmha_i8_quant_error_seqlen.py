import numpy as np
import matplotlib.pyplot as plt

SOFTMAX_QUANT_SCALE = 127
def quantize_to_int8(tensor, clip_max, quant_range=127):
    scale = quant_range / clip_max
    min_bound = - quant_range
    max_bound = quant_range
    outputs = np.clip((tensor.astype(np.float32) * scale).round(), min_bound, max_bound)
    quant_tensor = outputs.astype(np.int8)
    return quant_tensor


def quantize_to_uint8(tensor, clip_max, quant_range=255):  #
    scale = quant_range / clip_max
    max_bound = quant_range
    outputs = np.clip((tensor.astype(np.float32) * scale).round(), 0, max_bound)
    quant_tensor = outputs.astype(np.uint8)
    return quant_tensor

SEQLEN_LIST = [i*16 for i in range(1,65)]
print(SEQLEN_LIST)

HEAD_DIM = 64
error_o1_list = []
error_o2_list = []
error_o3_list = []
for SEQLEN in SEQLEN_LIST:
    f_q = np.random.randn(SEQLEN, HEAD_DIM)
    f_k = np.random.randn(SEQLEN, HEAD_DIM)
    f_v = np.random.randn(SEQLEN, HEAD_DIM)

    # GEMM1
    f_s = np.matmul(f_q, np.transpose(f_k), dtype = np.float32) / np.sqrt(HEAD_DIM)
    f_p = np.zeros_like(f_s)

    # Softmax
    for row in range(SEQLEN):
        row_max = np.max(f_s[row])  # scalar
        row_exp = np.exp(f_s[row] - row_max)  # vector
        row_exp_sum = np.sum(row_exp) # scalar
        f_p[row] = row_exp/row_exp_sum

    # GEMM2
    f_o_ref = np.matmul(f_p, f_v, dtype = np.float32)  # Ground truth

    # input quant arguments
    q_amax = np.abs(f_q).max()
    k_amax = np.abs(f_k).max()
    v_amax = np.abs(f_v).max()
    p_amax_ref = np.abs(f_p).max()  # potential
    o_amax = np.abs(f_o_ref).max()

    i8_q = quantize_to_int8(f_q, q_amax)
    i8_k = quantize_to_int8(f_k, k_amax)
    i8_v = quantize_to_int8(f_v, v_amax)

    # C1 GEMM1
    i32_s = np.matmul(i8_q, np.transpose(i8_k), dtype = np.int32)
    f_s = (i32_s * q_amax/127 * k_amax/127) / np.sqrt(HEAD_DIM)
    f_p = np.zeros_like(f_s, dtype=np.float32)

    # C1 Softmax
    p_amax = 1
    for row in range(SEQLEN):
        row_max = np.max(f_s[row])  # scalar
        row_exp = np.exp(f_s[row] - row_max)  # vector
        row_exp_sum = np.sum(row_exp) # scalar
        f_p[row] = row_exp/row_exp_sum

    i8_p = quantize_to_uint8(f_p, p_amax,SOFTMAX_QUANT_SCALE)
    # C1 GEMM2
    i32_o = np.matmul(i8_p, i8_v, dtype = np.int32)  # int8 result for case 1
    f_o1 = i32_o * p_amax/SOFTMAX_QUANT_SCALE * v_amax/127

    error_o1_list.append(np.sum(np.abs(f_o1 - f_o_ref)))

    # C2 GEMM1
    i32_s = np.matmul(i8_q, np.transpose(i8_k), dtype = np.int32)
    f_s = (i32_s * q_amax/127 * k_amax/127) / np.sqrt(HEAD_DIM)
    f_p = np.zeros_like(f_s, dtype=np.float32)

    # C2 Softmax
    p_amax = p_amax_ref
    for row in range(SEQLEN):
        row_max = np.max(f_s[row])  # scalar
        row_exp = np.exp(f_s[row] - row_max)  # vector
        row_exp_sum = np.sum(row_exp) # scalar
        f_p[row] = row_exp/row_exp_sum

    i8_p = quantize_to_uint8(f_p, p_amax,SOFTMAX_QUANT_SCALE)
    # C2 GEMM2
    i32_o = np.matmul(i8_p, i8_v, dtype = np.int32)  # int8 result for case 1
    f_o2 = i32_o * p_amax/SOFTMAX_QUANT_SCALE * v_amax/127

    error_o2_list.append(np.sum(np.abs(f_o2 - f_o_ref)))

    # C3 GEMM1
    i32_s = np.matmul(i8_q, np.transpose(i8_k), dtype = np.int32)
    f_s = (i32_s * q_amax/127 * k_amax/127) / np.sqrt(HEAD_DIM)
    f_p = np.zeros_like(f_s, dtype=np.float32)

    # C3 Softmax
    ui8_p = np.zeros_like(f_p, dtype=np.uint8)
    p_amax = np.zeros(SEQLEN, dtype=np.float32)
    for row in range(SEQLEN):
        row_max = np.max(f_s[row])  # scalar
        row_exp = np.exp(f_s[row] - row_max)  # vector
        row_exp_sum = np.sum(row_exp) # scalar
        f_p[row] = row_exp/row_exp_sum
        p_amax[row] = np.max(f_p[row])
        ui8_p[row] = quantize_to_uint8(f_p[row], p_amax[row],SOFTMAX_QUANT_SCALE)

    # C3 GEMM2
    i32_o = np.matmul(ui8_p, i8_v, dtype = np.int32)  # int8 result for case 1
    f_o3 = np.matmul(np.diag(p_amax/SOFTMAX_QUANT_SCALE), i32_o) * v_amax/127

    error_o3_list.append(np.sum(np.abs(f_o3 - f_o_ref)))  # np.linalg.norm(f_o3 - f_o_ref, ord=2, axis=None)

fig1, ax1 = plt.subplots()

# for SEQLEN in SEQLEN_LIST:
ax1.plot(SEQLEN_LIST, error_o1_list)
ax1.plot(SEQLEN_LIST, error_o2_list)
ax1.plot(SEQLEN_LIST, error_o3_list)

ax1.legend(["worst case", "static quantization", "dynamic quantization"])
ax1.set_ylabel("Error Sum")
ax1.set_xlabel("Input Sequence Length")
plt.show()
