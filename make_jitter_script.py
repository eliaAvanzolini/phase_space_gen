import re
import sys

with open("train_depth_cfm.py", "r") as f:
    code = f.read()

# 1. Aggiunta argomento --jitter al parser
if "--jitter" not in code:
    arg_code = '    parser.add_argument("--jitter", type=float, default=0.05, help="Std rumore gaussiano su profondita d in training")\n'
    # Inserisce prima del return del parser o prima di parse_args
    if "return parser" in code:
        code = code.replace("return parser", arg_code + "    return parser")
    elif "args = parser.parse_args()" in code:
        code = code.replace("args = parser.parse_args()", arg_code + "    args = parser.parse_args()")
    else:
        # Trova l'ultimo parser.add_argument
        matches = list(re.finditer(r'parser\.add_argument\([^)]+\)\n', code))
        if matches:
            idx = matches[-1].end()
            code = code[:idx] + arg_code + code[idx:]
        else:
            print("Errore: impossibile localizzare il blocco argparse.")
            sys.exit(1)

# 2. Iniezione del rumore nel training loop
# Cerca il pattern del DataLoader di training (es. for x, c in train_loader:)
loop_match = re.search(r'for\s+([a-zA-Z0-9_]+),\s*([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_]*train[a-zA-Z0-9_]*):', code)

if loop_match:
    feat_var, cond_var, loader_var = loop_match.groups()
    indent = " " * (loop_match.start() - code.rfind("\n", 0, loop_match.start()) - 1 + 4)
    injection = f"\n{indent}if getattr(args, 'jitter', 0.0) > 0:\n{indent}    {cond_var} = {cond_var} + torch.randn_like({cond_var}) * args.jitter"
    
    # Inserisce subito dopo la linea del 'for'
    insert_pos = loop_match.end()
    code = code[:insert_pos] + injection + code[insert_pos:]
    print(f"Jitter inserito con successo su tensore '{cond_var}' nel loop di '{loader_var}'.")
else:
    # Cerca pattern generico con batch (for batch in train_loader:)
    batch_match = re.search(r'for\s+([a-zA-Z0-9_]+)\s+in\s+([a-zA-Z0-9_]*train[a-zA-Z0-9_]*):', code)
    if batch_match:
        b_var, loader_var = batch_match.groups()
        indent = " " * (batch_match.start() - code.rfind("\n", 0, batch_match.start()) - 1 + 4)
        injection = f"\n{indent}# Depth Jittering\n{indent}# (verificare il nome della variabile condizione se unpackata sotto)"
        insert_pos = batch_match.end()
        code = code[:insert_pos] + injection + code[insert_pos:]
        print(f"Jitter predisposto nel loop '{loader_var}'.")
    else:
        print("Attenzione: pattern del loop non univoco. Verifica manuale consigliata.")

with open("train_depth_cfm_jitter.py", "w") as f:
    f.write(code)

print("Creato train_depth_cfm_jitter.py.")
