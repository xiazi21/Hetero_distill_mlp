from pathlib import Path

path = Path('two_teachjer_kd_update_han_direct_student_ts_v2.py')
text = path.read_text(encoding='utf-8')

old_total = "    parser.add_argument('--lambda_rel_total', type=float, default=None,\n                    help="
new_total = "    parser.add_argument('--lambda_rel_total', type=float, default=None,\n                        help="
old_balance = "    parser.add_argument('--relation_balance', type=float, default=None,\n                    help="
new_balance = "    parser.add_argument('--relation_balance', type=float, default=None,\n                        help="
text = text.replace(old_total, new_total)
text = text.replace(old_balance, new_balance)
path.write_text(text, encoding='utf-8')
