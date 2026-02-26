from src.scripts.run_dae import create_onehot_schema, apply_onehot_schema, run_dae
from src.scripts.run_missforest import run_missforest
from src.scripts.run_medianmode import run_medianmode
from src.scripts.run_meanmode import run_meanmode
from src.scripts.run_mice import run_mice

if __name__ == "__main__":
    run_meanmode()
    run_medianmode()
    run_mice()
    run_missforest()
    run_dae()

