from pathlib import Path
import re
import textwrap

path = Path(r'c:/Users/xiazi/Desktop/CursorProject/relation_only_hetero/two_teachjer_kd_update_han_direct_student_ts_v2.py')
text = path.read_text(encoding='utf-8')

meta_block = textwrap.dedent("""
def relation_combined_loss(
    rel_result: Union[torch.Tensor, Dict[str, Union[int, float, torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]]]],
    struct_logits: Optional[torch.Tensor],
    y: torch.Tensor,
    idx_train: torch.Tensor,
    lambda_rel_pos: float,
    lambda_rel_struct: float,
    lambda_rel_total: Optional[float],
    balance_override: Optional[float],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    def _to_tensor(val):
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return torch.tensor(float(val), device=device)

    if isinstance(rel_result, dict):
        if rel_result.get('relation_mean_tensors'):
            rel_core = torch.stack(list(rel_result['relation_mean_tensors'].values())).sum()
        else:
            rel_core = rel_result.get('total_loss', torch.tensor(0.0, device=device))
            if not isinstance(rel_core, torch.Tensor):
                rel_core = torch.tensor(float(rel_core), device=device)
    else:
        rel_core = _to_tensor(rel_result)

    if struct_logits is not None:
        struct_core = F.cross_entropy(struct_logits[idx_train], y[idx_train])
    else:
        struct_core = torch.tensor(0.0, device=device)

    rel_weight_raw = max(float(lambda_rel_pos), 0.0)
    struct_weight_raw = max(float(lambda_rel_struct), 0.0)

    if lambda_rel_total is not None:
        scale_value = float(lambda_rel_total)
    else:
        scale_value = rel_weight_raw + struct_weight_raw

    if scale_value <= 0.0:
        zero = torch.tensor(0.0, device=device)
        return {
            'total': zero,
            'scaled': {'relpos': zero, 'struct': zero},
            'weights': {
                'relpos': torch.tensor(0.0, device=device),
                'struct': torch.tensor(0.0, device=device),
                'scale': torch.tensor(0.0, device=device),
            },
            'raw': {'relpos': rel_core, 'struct': struct_core},
        }

    if balance_override is not None:
        balance = float(min(max(balance_override, 0.0), 1.0))
        struct_weight = balance
        rel_weight = 1.0 - balance
    else:
        total_raw = rel_weight_raw + struct_weight_raw
        if total_raw > 0.0:
            rel_weight = rel_weight_raw / total_raw
            struct_weight = struct_weight_raw / total_raw
        else:
            rel_weight = 0.5
            struct_weight = 0.5

    scale_tensor = torch.tensor(scale_value, device=device)
    rel_component = rel_core * rel_weight * scale_tensor
    struct_component = struct_core * struct_weight * scale_tensor
    total = rel_component + struct_component

    return {
        'total': total,
        'scaled': {'relpos': rel_component, 'struct': struct_component},
        'weights': {
            'relpos': torch.tensor(rel_weight, device=device),
            'struct': torch.tensor(struct_weight, device=device),
            'scale': scale_tensor,
        },
        'raw': {'relpos': rel_core, 'struct': struct_core},
    }


def meta_path_alignment_losses(
    mp_teacher: Dict[str, torch.Tensor],
    mp_student: Dict[str, torch.Tensor],
    tail_teacher: torch.Tensor,
    tail_student: torch.Tensor,
    teacher_proj: nn.Module,
    student_proj: nn.Module,
    beta_teacher: torch.Tensor,
    beta_student: torch.Tensor,
    reliability: torch.Tensor,
    metapath_keys: List[str],
    component_weights: Optional[Dict[str, float]] = None,
    eps: float = 1e-8,
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    device = reliability.device
    reliability = reliability.view(-1)
    active_keys = [key for key in metapath_keys if key in mp_teacher and key in mp_student]
    if len(active_keys) == 0:
        zero = torch.tensor(0.0, device=device)
        zero_w = torch.tensor(0.0, device=device)
        return {
            'feature': zero,
            'relpos': zero,
            'beta': zero,
            'total': zero,
            'scaled': {'feature': zero, 'relpos': zero, 'beta': zero},
            'raw': {'feature': zero, 'relpos': zero, 'beta': zero},
            'weights': {'feat': zero_w, 'relpos': zero_w, 'beta': zero_w},
            'details': {'active_keys': []},
        }

    num_nodes = reliability.size(0)
    teacher_tail = F.normalize(teacher_proj(tail_teacher), dim=-1)
    student_tail = F.normalize(student_proj(tail_student), dim=-1)

    teacher_idx = {key: idx for idx, key in enumerate(mp_teacher.keys())}
    student_idx = {key: idx for idx, key in enumerate(mp_student.keys())}
    teacher_fill_dtype = beta_teacher.dtype if beta_teacher.numel() > 0 else teacher_tail.dtype
    student_fill_dtype = beta_student.dtype if beta_student.numel() > 0 else student_tail.dtype

    def _align_beta(beta: Optional[torch.Tensor], index_map: Dict[str, int], fill_dtype: torch.dtype) -> torch.Tensor:
        if beta is None or beta.numel() == 0:
            aligned = torch.full((num_nodes, len(active_keys)), 1.0 / max(1, len(active_keys)),
                                 device=device, dtype=fill_dtype)
        else:
            cols = []
            for key in active_keys:
                idx = index_map.get(key, -1)
                if 0 <= idx < beta.size(1):
                    cols.append(beta[:, idx])
                else:
                    cols.append(torch.full((num_nodes,), 1.0 / max(1, len(active_keys)),
                                           device=device, dtype=beta.dtype))
            aligned = torch.stack(cols, dim=1)
        aligned = aligned.clamp_min(eps)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(eps)
        return aligned

    beta_teacher_aligned = _align_beta(beta_teacher, teacher_idx, teacher_fill_dtype)
    beta_student_aligned = _align_beta(beta_student, student_idx, student_fill_dtype)

    feature_terms = []
    relpos_terms = []
    path_gate_means = []
    rel_align_stack = []
    feat_align_stack = []

    identity_edges = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.stack([identity_edges, identity_edges], dim=0)

    for idx, key in enumerate(active_keys):
        teacher_mp = F.normalize(teacher_proj(mp_teacher[key]), dim=-1)
        student_mp = F.normalize(student_proj(mp_student[key]), dim=-1)

        rel_teacher = _edge_relpos(teacher_tail, teacher_mp, edge_index)
        rel_student = _edge_relpos(student_tail, student_mp, edge_index)
        rel_diff = (rel_teacher - rel_student).pow(2).sum(dim=-1)
        rel_align = torch.exp(-rel_diff)

        feat_diff = (teacher_mp - student_mp).pow(2).sum(dim=-1)
        feat_align = torch.exp(-feat_diff)

        gate_teacher = beta_teacher_aligned[:, idx]
        gate_student = beta_student_aligned[:, idx]
        gate = 0.5 * (gate_teacher + gate_student)
        base_gate = gate * reliability

        feature_weight = base_gate * rel_align
        rel_weight = base_gate

        feature_terms.append((feat_diff * feature_weight).mean())
        relpos_terms.append((rel_diff * rel_weight).mean())
        path_gate_means.append(base_gate.mean())
        rel_align_stack.append(rel_align)
        feat_align_stack.append(feat_align)

    feature_loss = torch.stack(feature_terms).mean()
    relpos_loss = torch.stack(relpos_terms).mean()
    rel_align_matrix = torch.stack(rel_align_stack, dim=1)
    feat_align_matrix = torch.stack(feat_align_stack, dim=1)

    rel_align_mean = rel_align_matrix.mean(dim=1)
    feat_align_mean = feat_align_matrix.mean(dim=1)
    alignment_for_beta = rel_align_mean * feat_align_mean

    beta_loss = meta_path_beta_loss(beta_teacher_aligned, beta_student_aligned, reliability, alignment_for_beta)
    attn_similarity = F.cosine_similarity(beta_teacher_aligned, beta_student_aligned, dim=1).mean()

    weights = component_weights or {}
    feat_w = float(weights.get('feat', 0.0))
    rel_w = float(weights.get('relpos', 0.0))
    beta_w = float(weights.get('beta', 0.0))

    scaled = {
        'feature': feature_loss * feat_w,
        'relpos': relpos_loss * rel_w,
        'beta': beta_loss * beta_w,
    }
    total = scaled['feature'] + scaled['relpos'] + scaled['beta']

    return {
        'feature': feature_loss,
        'relpos': relpos_loss,
        'beta': beta_loss,
        'total': total,
        'scaled': scaled,
        'raw': {
            'feature': feature_loss,
            'relpos': relpos_loss,
            'beta': beta_loss,
        },
        'weights': {
            'feat': torch.tensor(feat_w, device=device),
            'relpos': torch.tensor(rel_w, device=device),
            'beta': torch.tensor(beta_w, device=device),
        },
        'details': {
            'active_keys': active_keys,
            'attn_similarity': attn_similarity,
            'mean_gate': torch.stack(path_gate_means).mean(),
            'rel_align_mean': rel_align_mean.mean(),
            'feat_align_mean': feat_align_mean.mean(),
        },
    }


def meta_path_beta_loss(beta_teacher: torch.Tensor,
                        beta_student: torch.Tensor,
                        reliability: torch.Tensor,
                        alignment: Optional[torch.Tensor] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    if beta_teacher.numel() == 0 or beta_student.numel() == 0:
        return torch.tensor(0.0, device=reliability.device)
    log_student = beta_student.clamp_min(eps).log()
    loss = F.kl_div(log_student, beta_teacher, reduction='none').sum(dim=-1)
    weight = reliability
    if alignment is not None:
        weight = weight * alignment
    return (loss * weight).mean()
""").strip()

meta_start = text.index("def meta_path_feature_loss")
meta_end = text.index("def metapath2vec_category_embeddings")
text = text[:meta_start] + meta_block + "\n\n" + text[meta_end:]

training_block = textwrap.dedent("""
rel_out = relation_relative_pos_l2(
    taps_teacher=taps_projected,
    rel_student=rel_student,
    hetero=hetero,
    category=category,
    reliability=rho_r,
    projector_t=None,
    projector_s=None,
    relation_weights=None,
    return_details=True,
    include_per_edge=False)

struct_logits = student.structural_logits_direct(rel_base_aux, category)

relation_losses = relation_combined_loss(
    rel_result=rel_out,
    struct_logits=struct_logits,
    y=y,
    idx_train=idx_train,
    lambda_rel_pos=args.lambda_rel_pos,
    lambda_rel_struct=args.lambda_rel_struct,
    lambda_rel_total=getattr(args, 'lambda_rel_total', None),
    balance_override=getattr(args, 'relation_balance', None),
    device=device,
)

mp_student_embs = build_student_metapath_embs_direct(
    hetero=hetero,
    ops_template=ops_template,
    mp_base=mp_base_aux,
    category=category,
    device=device,
)

mp_keys = list(dict.fromkeys(list(mp_teacher_embs.keys()) + list(mp_student_embs.keys())))
if mp_student_embs:
    beta_student = student.meta_path_attention(mp_student_embs, mp_keys)
else:
    beta_student = torch.zeros((tail_student_aux.size(0), 0), device=device)

mp_losses = meta_path_alignment_losses(
    mp_teacher=mp_teacher_embs,
    mp_student=mp_student_embs,
    tail_teacher=tail_teacher,
    tail_student=tail_student_aux,
    teacher_proj=teacher_delta_proj,
    student_proj=student.delta_proj,
    beta_teacher=beta_teacher,
    beta_student=beta_student,
    reliability=rho_h,
    metapath_keys=mp_keys,
    component_weights={
        'feat': args.lambda_mp_feat,
        'relpos': args.lambda_mp_relpos,
        'beta': args.lambda_mp_beta,
    },
)

rel_loss = relation_losses['scaled']['relpos']
struct_loss = relation_losses['scaled']['struct']
relation_total = relation_losses['total']

mp_feat_loss = mp_losses['scaled']['feature']
mp_relpos = mp_losses['scaled']['relpos']
mp_beta = mp_losses['scaled']['beta']
mp_total = mp_losses['total']

loss = ce_loss + kd_loss + relation_total + mp_total
""").strip()

training_pattern = re.compile(r"        rel_out = relation_relative_pos_l2\([\s\S]+?loss = ce_loss \+ kd_loss \+ rel_loss \+ struct_loss \+ mp_feat_loss \+ mp_relpos \+ mp_beta")
replacement = textwrap.indent(training_block, "        ")
text, count = training_pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("Training block replacement failed")

arg_pattern = re.compile(r"    parser.add_argument\('--lambda_rel_struct', type=float, default=0\)\n")
arg_replacement = textwrap.dedent("""    parser.add_argument('--lambda_rel_struct', type=float, default=0)
    parser.add_argument('--lambda_rel_total', type=float, default=None,
                        help='Overall scaling for the combined relation loss (defaults to rel_pos + rel_struct).')
    parser.add_argument('--relation_balance', type=float, default=None,
                        help='Optional [0,1] ratio for the structural branch within the relation loss (1 => structure only).')
""")
text, count = arg_pattern.subn(arg_replacement, text, count=1)
if count != 1:
    raise SystemExit("Argument insertion failed")

path.write_text(text, encoding='utf-8')
