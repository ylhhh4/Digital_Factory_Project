# Digital_Factory

## Pilot Test

### To run

```bash
python vebjs_yolo.py veb2yolo 11106655_pallet_part00_phase20/pilot/M*.json -o 11106655_pallet_part00_phase20 --task both --rename K05=pallet_part00_phase20 K06=pallet_part00_phase20 K06_copy1=pallet_part00_phase20 K06_copy2=pallet_part00_phase20 K06_copy3=pallet_part00_phase20
```

### To check the distribution of the images

```bash
python check_dist.py
```

### How to clone objects in M2 mode

Modify the json file or use Ontology

My own link:

https://digital-factory.io/vebjs/?inputscene=https://raw.githubusercontent.com/ylhhh4/Digital_Factory_Project/main/FMS-02_M2.json&repoMod3d=https://murgo-polimi.github.io/VEB.js-FMS-00/models/FMS/GLB/

Professor:

https://digital-factory.io/vebjs/?inputscene=https://raw.githubusercontent.com/ylhhh4/Digital_Factory_Project/main/FMS-02_mod.json&repoMod3d=https://murgo-polimi.github.io/VEB.js-FMS-00/models/FMS/GLB/

What's the difference between the two json file?
The parentObject, no parentObject for the pallet when duplicating