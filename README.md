# FMHA-INT8-Quantization
## Abstract and Introduction
### Abstract
In this work, we quantize fused multi-head attention (FMHA) and Flash-Attention to lower precision 8-bit integers in the Transformer inference. The proposed method leverages the very nature of Softmax computation without requiring further prior knowledge of the input data. We improve the accuracy of the attention output of the fused kernel by about a factor of 2 in the simulation.

### Introduction
In this project, we aim to accelerate the FMHA mechanism during the 8-bit Transformer inference of language and vision models using Iluvatar GPGPU. Compared to FP32 inference, employing 8-bit integer (INT8 and UINT8) potentially consumes 4× less storage space but is up to 6× faster \cite{quinn2018pieces}. INT8 is 10× more en To adapt FP32 algorithms to INT8 algorithms, we need two techniques - quantization and dequantization \cite{gong2018highly}.

## Background: FMHA and FLash-Attention
### Attention 
\cite{vaswani2017attention, parikh2016decomposable, chaudhari2021attentive}
$$ Attention(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = Softmax\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V} $$
### Flash Attention
\cite{dao2022flashattention}
$$ \mathbf{S}_i = \mathbf{Q}\cdot\mathbf{K}^T_i $$
$$ \tilde{\mathbf{m}}_i = rowmax(\mathbf{S}_i) $$
$$ \tilde{\mathbf{M}}_i = diag{(\tilde{\mathbf{m}}_i)} $$
$$ \mathbf{P}_i = \exp{(\mathbf{S}_i-\tilde{\mathbf{M}}_i\cdot \mathbf{J}}) $$
$$ \tilde{\mathbf{l}}_i = rowsum(\mathbf{P}_i) $$
$$ \mathbf{m}_{i} = \max(\mathbf{m}_{i-1},\tilde{\mathbf{m}}_i) $$
$$ \mathbf{l}_{i} = \exp{(\mathbf{m}_{i-1}-\mathbf{m}_{i})}\cdot \mathbf{l}_{i-1} + \exp{(\tilde{\mathbf{m}}_i-\mathbf{m}_{i})}\cdot \tilde{\mathbf{l}}_i  $$
$$ \mathbf{M}_{i-1} = diag{(\mathbf{m}_{i-1})} $$
$$ \mathbf{M}_i = diag{(\mathbf{m}_{i})} $$
$$ \mathbf{L}_{i-1} = diag{(\mathbf{l}_{i-1})} $$
$$ \mathbf{L}_i = diag{(\mathbf{l}_{i})} $$
$$ \mathbf{O}_i =  \mathbf{L}_i^{-1} \cdot \left[ \mathbf{L}_{i-1} \cdot \exp{(\mathbf{M}_{i-1}-\mathbf{M}_{i})} \cdot  \mathbf{O}_{i-1} + \exp{(\tilde{\mathbf{M}}_i-\mathbf{M}_{i})}  \cdot \mathbf{P}_i \cdot\mathbf{V}_i \right] $$
where $\mathbf{Q} \in \mathbb{R}^{N_{src} \times d} $, $\mathbf{O}_i \in \mathbb{R}^{N_{src}\times d}$, $\mathbf{K}_i \in \mathbb{R}^{N_{trg}\times d}$, $\mathbf{V}_i \in \mathbb{R}^{N_{trg}\times d}$, $\mathbf{S}_i \in \mathbb{R}^{N_{src}\times N_{trg}}$, $\mathbf{P}_i \in \mathbb{R}^{N_{src}\times N_{trg}}$ are matrices and $\tilde{\mathbf{m}}_i \in \mathbb{R}^{N_{src}}$, $\tilde{\mathbf{l}}_i \in \mathbb{R}^{N_{src}}$, $\mathbf{m}_{i} \in \mathbb{R}^{N_{src}}$, $\mathbf{l}_{i} \in \mathbb{R}^{N_{src}}$ are vectors.

## 8-bit Quantized Attention
### 8-bit FMHA
$$ Attention(\mathbf{Q}_{\texttt{INT8}}, \mathbf{K}_{\texttt{INT8}}, \mathbf{V}_{\texttt{INT8}}) = \left\{ \left[ \left[ Softmax\left(\frac{\left( \mathbf{Q}_{\texttt{INT8}} \cdot \mathbf{K}^T_{\texttt{INT8}} \right)_{\texttt{INT32}}}{\sqrt{d}_{\texttt{FP32}}}\right)_{\texttt{FP32}} \right]_{\texttt{INT8}} \cdot \mathbf{V}_{\texttt{INT8}}\right]_{\texttt{INT32}}\right\}_{\texttt{INT8}} $$
See Figure \ref{fig:fmha_quant_diagram} 

$$\mathbf{S}_{\texttt{INT32}} = \mathbf{Q}_{\texttt{INT8}} \cdot \mathbf{K}^T_{\texttt{INT8}}$$
$$\mathbf{S}_{\texttt{FP32}} = \mathbf{S}_{\texttt{INT32}}\cdot \frac{1}{\sqrt{d}} \cdot\frac{\alpha_q}{127}\cdot\frac{\alpha_k}{127}$$
$$ \mathbf{m}_{\texttt{FP32}} = rowmax(\mathbf{S}_{\texttt{FP32}}) $$
$$ \mathbf{M}_{\texttt{FP32}} = diag(\mathbf{m}_{\texttt{FP32}}) $$
$$ \mathbf{P}_{\texttt{FP32}} = \exp{(\mathbf{S}_{\texttt{FP32}}-\mathbf{M}_{\texttt{FP32}}\cdot \mathbf{J})} $$
$$ \mathbf{l}_{\texttt{FP32}} = rowsum(\mathbf{P}_{\texttt{FP32}}) $$
$$ \mathbf{L}_{\texttt{FP32}} = diag(\mathbf{l}_{\texttt{FP32}}) $$
$$ \mathbf{P}_{\texttt{UINT8}} = \left[\left( \frac{\mathbf{L}_{\texttt{FP32}}^{-1}}{255} \right)^{-1} \cdot \mathbf{L}_{\texttt{FP32}}^{-1} \cdot \mathbf{P}_{\texttt{FP32}}\right]_{0}^{255} = \left[255 \cdot \mathbf{P}_{\texttt{FP32}}\right]_{0}^{255}  $$
$$ \mathbf{O}_{\texttt{INT32}} = \mathbf{P}_{\texttt{UINT8}} \cdot \mathbf{V}_{\texttt{INT8}} $$
$$ \mathbf{O}_{\texttt{FP32}} = \frac{\mathbf{L}_{\texttt{FP32}}}{255} \cdot \frac{\alpha_v}{127} \cdot \mathbf{O}_{\texttt{INT32}} $$
$$ \mathbf{O}_{\texttt{INT8}} = \left[\frac{127}{\alpha_o} \cdot \mathbf{O}_{\texttt{FP32}}\right]_{-127}^{127} $$
$\mathbf{J}$ is matrix of ones in $\mathbb{R}^{N_{src}\times N_{trg}}$. $\alpha_q$, $\alpha_k$, $\alpha_v$ and $\alpha_o$ represent respectively the predetermined maximum absolute value of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ and $\mathbf{O}$.

### 8-bit Flash-Attention
$$ \mathbf{S}_{\texttt{INT32},i} = \mathbf{Q}_{\texttt{INT8}}\cdot\mathbf{K}^T_{\texttt{INT8},i} $$
$$ \mathbf{S}_{\texttt{FP32},i} = \mathbf{S}_{\texttt{INT32},i}\cdot \frac{1}{\sqrt{d}} \cdot\frac{\alpha_q}{127}\cdot\frac{\alpha_k}{127}$$
$$ \tilde{\mathbf{m}}_{\texttt{FP32},i} = rowmax(\mathbf{S}_{\texttt{FP32},i}) $$
$$ \tilde{\mathbf{M}}_{\texttt{FP32},i} = diag{(\tilde{\mathbf{m}}_{\texttt{FP32},i})} $$
$$ \mathbf{P}_{\texttt{FP32},i} = \exp{(\mathbf{S}_{\texttt{FP32},i}-\tilde{\mathbf{M}}_{\texttt{FP32},i}\cdot\mathbf{J})} $$
$$ \tilde{\mathbf{l}}_{\texttt{FP32},i} = rowsum(\mathbf{P}_{\texttt{FP32},i}) $$
$$ \tilde{\mathbf{L}}_{\texttt{FP32},i} = diag{(\tilde{\mathbf{l}}_{\texttt{FP32},i})} $$
$$ \mathbf{P}_{\texttt{UINT8},i} = \left[\left( \frac{\tilde{\mathbf{L}}_{\texttt{FP32,i}}^{-1}}{255} \right)^{-1} \cdot \tilde{\mathbf{L}}_{\texttt{FP32,i}}^{-1} \cdot \mathbf{P}_{\texttt{FP32},i}\right]_{0}^{255}=\left[255 \cdot \mathbf{P}_{\texttt{FP32},i}\right]_{0}^{255}$$
$$ \tilde{\mathbf{O}}_{\texttt{INT32},i} = \mathbf{P}_{\texttt{UINT8},i} \cdot\mathbf{V}_{\texttt{INT8},i} $$
$$ \tilde{\mathbf{O}}_{\texttt{FP32},i} =  \frac{\tilde{\mathbf{L}}_{\texttt{FP32},i}^{-1}}{255} \cdot \frac{\alpha_v}{127} \cdot \tilde{\mathbf{O}}_{\texttt{INT32},i} $$
$$ \mathbf{m}_{\texttt{FP32},i} = \max(\mathbf{m}_{\texttt{FP32},i-1},\tilde{\mathbf{m}}_{\texttt{FP32},i}) $$
$$ \mathbf{l}_{\texttt{FP32},i} = \exp{(\mathbf{m}_{\texttt{FP32},i-1}-\mathbf{m}_{\texttt{FP32},i})}\cdot \mathbf{l}_{\texttt{FP32},i-1} + \exp{(\tilde{\mathbf{m}}_{\texttt{FP32},i}-\mathbf{m}_{\texttt{FP32},i})}\cdot \tilde{\mathbf{l}}_{\texttt{FP32},i}  $$
$$ \mathbf{M}_{\texttt{FP32},i} = diag{(\mathbf{m}_{\texttt{FP32},i-1})} $$
$$ \mathbf{M}_{\texttt{FP32},i} = diag{(\mathbf{m}_{\texttt{FP32},i})} $$
$$ \mathbf{L}_{\texttt{FP32},i-1} = diag{(\mathbf{l}_{\texttt{FP32},i-1})} $$
$$ \mathbf{L}_{\texttt{FP32},i} = diag{(\mathbf{l}_{\texttt{FP32},i})} $$
$$ \mathbf{O}_{\texttt{FP32},i} =  \mathbf{L}_{\texttt{FP32},i}^{-1} \cdot \left[ \mathbf{L}_{\texttt{FP32},i-1} \cdot \exp{(\mathbf{M}_{\texttt{FP32},i-1}-\mathbf{M}_{\texttt{FP32},i})} \cdot  \mathbf{O}_{\texttt{FP32},i-1} + \tilde{\mathbf{L}}_{\texttt{FP32,i}} \cdot \exp{(\tilde{\mathbf{M}}_{\texttt{FP32},i}-\mathbf{M}_{\texttt{FP32},i})}  \cdot \tilde{\mathbf{O}}_{\texttt{FP32},i} \right] $$
$$ \mathbf{O}_{\texttt{INT8}, N} = \left[ \frac{127}{\alpha_o} \cdot \mathbf{O}_{\texttt{FP32}} \right]_{-127}^{127} $$

\begin{figure}
\centering
\includegraphics[width=0.8\textwidth]{fig/fmha_quant_diagram.png}
\caption{\label{fig:fmha_quant_diagram} The 8-bit quantization schematic diagram of the forward FMHA.}
\end{figure}

For Iluvatar MR GPGPU card, we only have tensor core unit (TCU) with input matrix of identical data type. So we will lose half of the quatization range during in the second GEMM resulting in a loss of precision.

## Result
\begin{figure}
\centering
\includegraphics[width=0.8\textwidth]{fig/deviation_512.png}
\caption{\label{fig:deviation_512} Deviation between the 8-bit quantization output and the groudtruth (FP32 reference). The worst case occurs when the quantization parameter $\alpha_p$ is chosen to $1$. Static quantization uses the predetermined maximum possible value of matrix $\mathbf{P}$.}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.8\textwidth]{fig/error_sum.png}
\caption{\label{fig:error_sum} The 8-bit quantization schematic diagram of the forward FMHA.}
\end{figure}

Run python simulation on FMHA to show deviation Figure \ref{fig:deviation_512}.

Run python simulation on FMHA to show the error summation of the output when increasing the sequence length \ref{fig:error_sum}.

Run BERT and real models with accuracy comparison .


Table~\ref{tab:widgets}

\begin{table}
\centering
\begin{tabular}{l|r|r|r|r}
Model Precision & BERT BASE 256  & BERT BASE 384 & BERT LARGE 256 & BERT LARGE 384 \\\hline
FP32 Baseline & & & & 89.97 \\
Static 8-bit & 87.163 & 87.433 & 88.765 & 89.787\\
Dynamic 8-bit & 87.154 & 87.526 & 88.704 & 89.861
\end{tabular}
\caption{\label{tab:widgets}The table lists the achieved F1 Scores of the BERT model during IXRT inference.}
\end{table}


\begin{table}
\centering
\begin{tabular}{l|r|r|r|r}
Model Precision & BERT BASE 256  & BERT BASE 384 & BERT LARGE 256 & BERT LARGE 384 \\\hline
FP32 Baseline & & & &  \\
Static 8-bit  & 79.990 & 80.123 & 82.034 & 82.800\\
Dynamic 8-bit  & 79.962 & 80.321 & 81.939 & 82.838\\
\end{tabular}
\caption{\label{tab:widgets}The table lists the achieved exact matches of the BERT model during IXRT inference.}
\end{table}

In practice, making the quantization factor greater while fixing the de-quantization factor can improve the two scores a little bit since it can amplify the elements $\mathbf{P}$ with values thus effectively pick out the values in $\mathbf{V}$ with higher possibility.
