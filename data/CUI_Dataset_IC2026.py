import json
import csv

# =========================
# Load CUI per image
# =========================
def load_concepts(concepts_path):
    image_to_cuis = {}

    with open(concepts_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            image_id = row[0].strip().replace('"', '')
            cuis_raw = row[1].strip().replace('"', '')

            cuis = cuis_raw.split(";") if cuis_raw else []
            image_to_cuis[image_id] = cuis

    return image_to_cuis


# =========================
# Load CUI → name mapping
# =========================
def load_cui_mapping(mapping_path):
    cui_to_name = {}

    with open(mapping_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cui = row["CUI"].strip()
            name = row["Canonical name"].strip()
            cui_to_name[cui] = name

    return cui_to_name


# =========================
# Enrich dataset (STRUCTURED)
# =========================
def enrich_dataset(json_path, concepts_dict, cui_mapping, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched = []

    for sample in data:
        image_id = sample["image_id"]

        cuis = concepts_dict.get(image_id, [])

        # Build structured list
        structured_cuis = []
        for cui in cuis:
            structured_cuis.append({
                "id": cui,
                "name": cui_mapping.get(cui, "UNKNOWN")
            })

        sample["cui"] = structured_cuis

        enriched.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print(f"Saved enriched dataset to {output_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    concepts_path = "dev_caption/concepts.csv"
    mapping_path = "dev_caption/cui_mapping.csv"

    train_json = "artifacts/datasets/imageclef2026_train_dataset.json"
    valid_json = "artifacts/datasets/imageclef2026_valid_dataset.json"

    output_train = "artifacts/datasets/IC2026_Caption_CUI_train.json"
    output_valid = "artifacts/datasets/IC2026_Caption_CUI_valid.json"

    concepts_dict = load_concepts(concepts_path)
    cui_mapping = load_cui_mapping(mapping_path)

    enrich_dataset(train_json, concepts_dict, cui_mapping, output_train)
    enrich_dataset(valid_json, concepts_dict, cui_mapping, output_valid)