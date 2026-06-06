# Source from the repo root:  source env.sh
# Puts both the landmark package (repo root) and the eval harness (eval/) on PYTHONPATH.
#   - repo root  -> lets `import landmark_baseline` resolve
#   - eval/      -> lets the harness find asl_backends, pytorch_i3d, custom_models, videotransforms, ...
VH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export VH_ROOT
export PYTHONPATH="$VH_ROOT:$VH_ROOT/eval:$PYTHONPATH"
echo "[visionary-hands] VH_ROOT=$VH_ROOT"
echo "[visionary-hands] PYTHONPATH set (repo root + eval/)"
