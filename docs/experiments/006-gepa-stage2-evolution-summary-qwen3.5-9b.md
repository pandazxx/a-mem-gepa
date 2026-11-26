| category | n | baseline F1 | candidate F1 | delta F1 [95% CI] | baseline BLEU-1 | candidate BLEU-1 |
|---|---|---|---|---|---|---|
| multi_hop | 83 | 0.2704 | 0.2748 | +0.0044 [-0.0628, +0.0726] | 0.2048 | 0.2205 |
| temporal | 103 | 0.3284 | 0.3683 | +0.0399 [-0.0105, +0.0921] | 0.2389 | 0.2766 |
| open_domain | 30 | 0.1663 | 0.1516 | -0.0147 [-0.0962, +0.0808] | 0.1505 | 0.1418 |
| single_hop | 250 | 0.4358 | 0.4670 | +0.0312 [-0.0159, +0.0765] | 0.3757 | 0.4091 |
| adversarial | 130 | 0.2990 | 0.3743 | +0.0752 [-0.0059, +0.1575] | 0.2801 | 0.3519 |
| overall | 596 | 0.3508 | 0.3871 | +0.0363 [+0.0070, +0.0662] | 0.2961 | 0.3340 |

Adversarial, refusal reading (picked 'Not mentioned'): baseline 0.7692, candidate 0.6846, delta -0.0846 [-0.1692, +0.0000]. The F1 row above is the paper's reading (scored against the trap text).
