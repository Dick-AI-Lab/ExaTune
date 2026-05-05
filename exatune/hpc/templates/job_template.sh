#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --output={{ log_dir }}/{{ job_name }}_%j.out
#SBATCH --error={{ log_dir }}/{{ job_name }}_%j.err
{% if partition %}#SBATCH --partition={{ partition }}{% endif %}
#SBATCH --time={{ time }}
#SBATCH --mem={{ memory }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --nodes={{ nodes }}
{% if account %}#SBATCH --account={{ account }}{% endif %}
{% if qos %}#SBATCH --qos={{ qos }}{% endif %}
# {% if email %}#SBATCH --mail-user={{ email }}{% endif %}
# {% if email_type %}#SBATCH --mail-type={{ email_type }}{% endif %}
{% for key, value in additional_directives.items() %}
#SBATCH --{{ key }}={{ value }}
{% endfor %}

# ExaTune Hyperparameter Search Job
# Job ID: {{ job_id }}
# Configuration Hash: {{ config_hash }}

echo "=================================================="
echo "ExaTune Job: {{ job_name }}"
echo "Started at: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

# Load environment modules
{% if modules %}
{% for module in modules %}
module load {{ module }}
{% endfor %}
{% endif %}

# Activate Python environment
{% if python_env %}
source {{ python_env }}/bin/activate
{% else %}
# Default: assume Python is in PATH
{% endif %}

# Set random seed for reproducibility
export PYTHONHASHSEED={{ random_seed }}

# Change to working directory
cd {{ work_dir }}

# Run the training script
python -u {{ script_path }} \
    --config {{ config_path }} \
    --job-id {{ job_id }} \
    --output-dir {{ output_dir }} \
    --hyperparameters '{{ hyperparameters_json }}' \
    {% if random_seed %}--random-seed {{ random_seed }}{% endif %}

EXIT_CODE=$?

echo "=================================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
