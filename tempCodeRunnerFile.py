import modularity_optuna
from optuna.study import create_study

study = create_study(direction="maximize", study_name=f"tgcntuning")
print('Study created successfully.')