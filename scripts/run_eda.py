import sys
sys.path.insert(0, 'src')
from pathlib import Path
from colony_counting.data.agar import load_agar_records
from colony_counting.data.eda import write_eda

records = load_agar_records(Path('Data/archive/data'))
summary = write_eda(records, Path('results/eda'))
print(f"EDA done: {summary['image_count']} images, {summary['annotated_object_count']} annotations")
print(f"Status distribution: {summary['status_distribution']}")
print(f"Species: {summary['species_annotation_distribution']}")
