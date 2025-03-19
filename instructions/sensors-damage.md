Here’s the reformatted version of your text for easy copying:

---

### **Table 3: Sensor Locations and Descriptions**

| Sensor Label | Description                   | Sensor Model | Units in Data File |
| ------------ | ----------------------------- | ------------ | ------------------ |
| AN3          | Ring gear radial 6 o’clock    | IMI 626B02   | m/s²               |
| AN4          | Ring gear radial 12 o’clock   | IMI 626B02   | m/s²               |
| AN5          | LS-SH radial                  | IMI 626B02   | m/s²               |
| AN6          | IMS-SH radial                 | IMI 622B01   | m/s²               |
| AN7          | HS-SH radial                  | IMI 622B01   | m/s²               |
| AN8          | HS-SH upwind bearing radial   | IMI 622B01   | m/s²               |
| AN9          | HS-SH downwind bearing radial | IMI 622B01   | m/s²               |
| AN10         | Carrier downwind radial       | IMI 622B01   | m/s²               |
| Speed\*      | HS-SH                         | IMI 626B02   | rpm                |

> \*Format is not the same for data collected from the “healthy” test gearbox.

---

---

### **3.5 Data File Description**

The data files are provided in the following format:

1. **Matlab packed binary format** (\*.mat) for direct import into Matlab. Header information identifies signals (variables). File converters/importers are readily available for large data handling.
2. **Ten 1-minute data sets** for the test condition described in Table 4. Files are labeled, e.g., `H1.mat` for the first minute of test data on the “healthy” gearbox.
3. **40 kHz data** provided in 10 one-dimensional arrays (no time channel included).

---

### **4. Actual Gearbox Damage**

The **"damaged" gearbox** experienced two oil-loss events in the field. It was later disassembled, and a detailed failure analysis was conducted [3].  
Table 5 summarizes the actual damage detected through vibration analysis.

---

### **Table 5: Actual Gearbox Damage Deemed Detectable through Vibration Analysis**

| Damage # | Component                        | Mode                                         |
| -------- | -------------------------------- | -------------------------------------------- |
| 1        | HS-ST gear set                   | Scuffing                                     |
| 2        | HS-SH downwind bearings          | Overheating                                  |
| 3        | IMS-ST gear set                  | Fretting corrosion, scuffing, polishing wear |
| 4        | IMS-SH upwind bearing            | Assembly damage, scuffing, dents             |
| 5        | IMS-SH downwind bearings         | Assembly damage, dents                       |
| 6        | Annulus/ring gear, or sun pinion | Scuffing and polishing, fretting corrosion   |
| 7        | Planet carrier upwind bearing    | Fretting corrosion                           |

---
