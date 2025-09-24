==========================================================================================================================================

        --========  Review Reports  ========--

The review report from reviewer #1:

*1: Is the paper relevant to Workshop?
  [_] No
  [X] Yes

*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [_] -2 (Marginally)
  [X] -4 (Not very much)
  [_] -6 (Not at all)

*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [_] 3 (High)
  [_] -2 (Marginal)
  [X] -4 (Low)
  [_] -6 (Very low)

*4: How is the presentation?
  [_] 6 (Excellent)
  [_] 3 (Good)
  [X] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)

*5: Is the paper of interest to Workshop users and practitioners?
  [_] 3 (Yes)
  [X] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)

*6: What is your confidence in your review of this paper?
  [X] 2 (High)
  [_] 1 (Medium)
  [_] 0 (Low)

*7: Overall recommendation
  [_] 5 (Strong Accept: top quality)
  [_] 4 (Accept: a regular paper)
  [_] 3 (Weak Accept: could be a poster)
  [X] 2 (Weak Reject: don't like it, but won't argue to reject it)
  [_] 1 (Reject: will argue to reject it)
  [_] 0 (Strong Reject: hopeless)

*8: Detailed comments for the authors
  This paper presents MM-GPT2Rec, combining GPT-2 (very old multi-modal model) with multimodal features for sequential recommendation. While addressing multimodal recommendation, the work has significant methodological and evaluation issues.

Critical Issues:
1. Lack of Proper Multimodal Baselines: The paper fails to compare against established multimodal recommendation methods. Key missing baselines include:

- VBPR (Visual Bayesian Personalized Ranking) - explicitly mentioned but not implemented
- DeepCoNN or NRMF for text-based recommendations
- CAMRec (mentioned in related work but not compared)

Recent multimodal sequential recommenders like MMGCN (https://github.com/weiyinwei/MMGCN) or SLMRec (https://arxiv.org/abs/2405.17890)
This is a fundamental flaw as it's impossible to assess whether the multimodal aspects provide genuine value.

Evaluation Credibility Issues:

1. Perfect scores (HR@10 = 1.0, MRR = 1.0) are unrealistic for real recommendation tasks, it might be due to the toy Amazon dataset, these results suggest either trivial data or flawed evaluation methodology
2. Single domain (Appliances only) severely limits generalizability claims
3. No statistical significance testing despite extraordinary performance claims

Technical Limitations:

1. Using outdated GPT-2 medium (345M parameters) while claiming to advance LLM-based recommendation
2. Simple concatenation for multimodal fusion without exploring sophisticated fusion techniques
3. No ablation study to understand individual modality contributions in this case text only, image only
4. Missing computational cost analysis for practical deployment

Experimental Design Flaws:

1. Extremely limited scale undermines broader applicability claims
2. No comparison with modern neural recommendation baselines (SASRec, BERT4Rec and others)
3. Baseline implementations appear suboptimal without implementation details
4. No validation methodology described, how many times this algorithm tested and what are the evaluation metrics? Reasoning would be great

Missing Critical Evaluations:
Given that this is a multimodal system, the paper should evaluate:

1. Modality Ablation: Performance with text-only vs image-only vs combined
2. Multimodal Fusion Comparison: Systematic comparison of fusion strategies
3. Cold-start Analysis: How well does content help with new items/users, whiuch is a major problem in real world setting
4. Domain Transfer: Performance across different product categories

Methodological Concerns:

1. I believe training details are insufficient for reproducibility
2. No discussion of hyperparameter sensitivity, like temperature, ranbdomness etc
3. Unclear handling of sequence length variations

Presentation Issues:

1. Figures don't clearly illustrate the multimodal architecture
2. Mathematical formulations lack precision


Fundamental Questions:
The perfect evaluation scores raise serious concerns about the experimental setup might be ok for toy dataset. Real recommendation systems rarely achieve such scores, suggesting either:

- The dataset/task is too simple to be meaningful
- The evaluation methodology has significant flaws
- The baselines are poorly implemented
- Need to compare with different multimodal models such as GPT4-o, Gemini nano,
- Would be great to have a comparision with SLM (small language multimodal models) if time permists

Recommendations for Major Revision:

1. Include proper multimodal recommendation baselines (VBPR, multimodal neural CF variants)
2. Conduct rigorous ablation studies on modality contributions
3. Expand to multiple product domains with realistic evaluation scores
4. Provide detailed computational cost analysis
5. Include modern neural recommendation baselines. GPT 2 is pretty old, running experiments on the latest multimodal models would enhance the results and generalization
5. Address the credibility issues with perfect evaluation scores by running multiple times of the same algorithm and oin different MMM models
6. Provide comprehensive implementation details for reproducibility

The core idea has merit, but the execution and evaluation need fundamental improvements before this work can contribute meaningfully to multimodal recommendation research.

========================================================
The review report from reviewer #2:

*1: Is the paper relevant to Workshop?
  [_] No
  [X] Yes

*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)

*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [_] 3 (High)
  [X] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)

*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)

*5: Is the paper of interest to Workshop users and practitioners?
  [_] 3 (Yes)
  [X] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)

*6: What is your confidence in your review of this paper?
  [X] 2 (High)
  [_] 1 (Medium)
  [_] 0 (Low)

*7: Overall recommendation
  [_] 5 (Strong Accept: top quality)
  [_] 4 (Accept: a regular paper)
  [_] 3 (Weak Accept: could be a poster)
  [X] 2 (Weak Reject: don't like it, but won't argue to reject it)
  [_] 1 (Reject: will argue to reject it)
  [_] 0 (Strong Reject: hopeless)

*8: Detailed comments for the authors
  Paper presents MM-GPT2Rec, a multimodal sequential recommender that fine-tunes GPT-2 for next-item prediction by combining textual reviews and visual product features. While the multimodal approach is reasonable and the presentation is clear, several significant limitations constrain the contribution's impact.

The experimental evaluation is severely limited in scope - only 40,000 interactions from a single Amazon category (Appliances) as acknowledged by the authors. This scale is insufficient to draw meaningful conclusions about LLM-based recommendation approaches, particularly when production systems handle millions of users and items. The perfect scores achieved (HR@10=1.0, MRR=1.0) raise concerns about experimental rigor, especially without statistical significance testing or detailed methodology for data splitting.

The user modeling approach reveals fundamental limitations. The authors explicitly state they removed user embeddings after finding no improvement, relying solely on item sequences to capture user preferences. They acknowledge their model "cannot directly help" with cold-start users due to its sequence-based nature. This creates a mismatch where a complex LLM architecture processes the same sparse behavioral signals that simpler sequential models could handle, without leveraging the rich contextual understanding that typically justifies LLM usage.

Technical novelty is incremental, essentially adapting existing transformer architectures for recommendation by treating items as tokens. The authors' constraint to GPT-2 medium due to "GPU memory limitations" questions practical viability. Baseline comparisons rely on classical methods (collaborative filtering, popularity-based) rather than modern neural approaches like SASRec or BERT4Rec that would provide more meaningful evaluation.

The experimental design lacks important details about train/test splits and potential data leakage. No ablation studies separate the contributions of text versus image features, and computational costs aren't analyzed compared to simpler alternatives. The authors honestly discuss limitations including dataset size and domain constraints, but these acknowledgments highlight the preliminary nature of the work.

While the paper addresses a relevant problem and demonstrates competent engineering, it falls short of making a convincing case for when LLM-based approaches provide genuine advantages over established methods. The limited scale, sparse user modeling, and lack of rigorous evaluation prevent drawing meaningful conclusions about the viability of this approach for practical recommendation systems.

========================================================
The review report from reviewer #3:

*1: Is the paper relevant to Workshop?
  [_] No
  [X] Yes

*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [X] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)

*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [X] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)

*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)

*5: Is the paper of interest to Workshop users and practitioners?
  [_] 3 (Yes)
  [X] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)

*6: What is your confidence in your review of this paper?
  [X] 2 (High)
  [_] 1 (Medium)
  [_] 0 (Low)

*7: Overall recommendation
  [_] 5 (Strong Accept: top quality)
  [_] 4 (Accept: a regular paper)
  [X] 3 (Weak Accept: could be a poster)
  [_] 2 (Weak Reject: don't like it, but won't argue to reject it)
  [_] 1 (Reject: will argue to reject it)
  [_] 0 (Strong Reject: hopeless)

*8: Detailed comments for the authors
  This paper introduces a multimodal sequential recommender system that adapts GPT-2 to predict users’ next likely purchases. By combining product review text and images into unified embeddings, the model captures richer user–item relationships and outperforms collaborative filtering, matrix factorization, and hybrid baselines on Amazon product data. The results highlight improvements in accuracy, catalog coverage, and diversity, while also discussing trade-offs with novelty.
The paper’s strengths lie in both its methodological design and empirical validation. By unifying textual reviews and product images into joint embeddings, the proposed model captures fine-grained semantic and visual signals that traditional CF or content-only baselines often miss, enabling more accurate next-item predictions. The adaptation of GPT-2 for sequential recommendation is technically interesting, treating purchase histories analogously to language sequences and thereby leveraging the transformer’s ability to model long-range dependencies in user behavior. The evaluation is thorough, with comparisons against multiple representative baselines across both accuracy-focused and beyond-accuracy metrics, and the results convincingly demonstrate gains in hit rate, NDCG, and catalog coverage, showing that the multimodal LLM approach provides practical benefits for real-world e-commerce recommendation scenarios.
One limitation is that the model relies on a relatively small and outdated backbone (GPT-2 medium) and a narrow slice of the Amazon dataset, which raises concerns about scalability and generalizability to larger, more diverse e-commerce domains. In addition, the paper does not include comparisons with more recent transformer-based recommenders, which makes it harder to position the contribution within the latest state of the art. Finally, while beyond-accuracy metrics are reported, the analysis of trade-offs such as novelty and fairness remains limited, and the work could benefit from more systematic exploration of these aspects.

========================================================