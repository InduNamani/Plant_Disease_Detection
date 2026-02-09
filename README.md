# Plant Disease Detection - Half Day Project Guide 🌿

## Project Overview
Build a simple plant disease detector using Machine Learning in ~4-5 hours. You'll learn feature extraction, ML algorithms, and deployment!

**Tech Stack:** Python + Scikit-learn + Streamlit + Supabase

---

## Phase 1: Setup (30 minutes)

### Step 1.1: Install Required Libraries
```bash
pip install streamlit scikit-learn opencv-python pillow pandas numpy supabase
pip install python-dotenv matplotlib seaborn
```

### Step 1.2: Project Folder Structure
Create this folder structure:
```
plant-disease-detection/
│
├── app.py                 # Streamlit app (frontend)
├── train_model.py         # ML model training
├── utils.py              # Helper functions
├── .env                  # Supabase credentials (secret)
├── requirements.txt      # Dependencies
│
├── data/
│   └── plantvillage/     # Dataset (download here)
│
└── models/
    └── plant_model.pkl   # Saved trained model
```

### Step 1.3: Download Dataset (Simplified)
We'll use a **small subset** for half-day completion:

**Option A - Quick Start (Recommended):**
Download only 5 classes from Kaggle:
- URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Download only these folders:
  - Tomato___Bacterial_spot
  - Tomato___Early_blight
  - Tomato___Late_blight
  - Tomato___Leaf_Mold
  - Tomato___healthy

Place them in `data/plantvillage/` folder.

**Why tomato?** Most images, clear disease patterns, practical use case.

---

## Phase 2: Understanding Feature Extraction (15 minutes - READ THIS!)

### What's Happening?
ML algorithms can't work with images directly. We need to convert images into **numbers (features)**.

### Features We'll Extract:

1. **Color Features:**
   - Mean RGB values (how much red, green, blue on average)
   - Color histogram (distribution of colors)
   - **Why?** Diseased leaves often have different colors

2. **Texture Features (Simple):**
   - Standard deviation of pixel values
   - Edge density using Canny edge detection
   - **Why?** Diseases create spots, patches, different textures

3. **Shape Features:**
   - Image dimensions
   - Aspect ratio
   - **Why?** Just basic metadata

### Example:
- Healthy leaf: High green values, low texture variation
- Diseased leaf: Brown/yellow patches, high texture variation

---

## Phase 3: Code Implementation (2.5 hours)

### File 1: `utils.py` - Feature Extraction Helper


### File 2: `train_model.py` - Train the ML Model

`
### File 3: `app.py` - Streamlit Frontend + Supabase



# ==================== STREAMLIT UI ====================



**What's happening here?**
- Streamlit creates the web interface
- User uploads image → extract features → predict using trained model
- Shows disease name, confidence, treatment info
- Saves to Supabase database

---

## Phase 4: Supabase Setup (20 minutes)

### Step 4.1: Create Supabase Account
1. Go to https://supabase.com
2. Sign up (it's free!)
3. Create a new project
4. Wait 2-3 minutes for setup

### Step 4.2: Create Database Table

Go to **SQL Editor** in Supabase dashboard and run:

```sql
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    disease_name TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    user_name TEXT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Enable Row Level Security (optional for this project)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow anyone to insert and read (for demo purposes)
CREATE POLICY "Enable insert for all" ON predictions FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable read for all" ON predictions FOR SELECT USING (true);
```

### Step 4.3: Get API Credentials

1. Go to **Project Settings** → **API**
2. Copy these two values:
   - Project URL
   - `anon` `public` key

### Step 4.4: Create `.env` File

Create `.env` in your project root:

```env
SUPABASE_URL=your_project_url_here
SUPABASE_KEY=your_anon_key_here
```

**⚠️ IMPORTANT:** Add `.env` to `.gitignore` to keep credentials secret!

---

## Phase 5: Run the Project (1 hour)

### Step 5.1: Train the Model

```bash
python train_model.py
```

**Expected output:**
```
Processing Tomato___Bacterial_spot: 100 images
Processing Tomato___Early_blight: 100 images
...
✓ Model Accuracy: 92.50%
✓ Model saved!
```

**What's happening?**
- Loading all images
- Extracting 107 features per image
- Training Random Forest with 100 trees
- Saving model to `models/` folder

Time: ~5-10 minutes

### Step 5.2: Run Streamlit App

```bash
streamlit run app.py
```

Browser will auto-open at `http://localhost:8501`

### Step 5.3: Test the App

1. Upload a tomato leaf image
2. Click "Detect Disease"
3. See prediction + confidence + treatment info
4. Check Supabase dashboard → you'll see new row in `predictions` table!

---

## Phase 6: Understanding What You Built (10 minutes)

### The ML Pipeline:

```
Image → Feature Extraction → ML Model → Prediction
```

**1. Feature Extraction (utils.py)**
- Converts 128x128 image (49,152 pixels) into 107 numbers
- These numbers describe color, texture, edges
- ML can understand numbers, not pixels directly

**2. Random Forest Algorithm**
- Creates 100 decision trees
- Each tree learns different patterns
- Final prediction = majority vote of all trees
- **Why Random Forest?** Accurate, fast, handles our 107 features well

**3. Training Process**
- Model learns patterns: "If green_mean < 100 AND texture > 50 → Bacterial spot"
- Uses 80% data to learn, 20% to test
- Accuracy ~85-95% (depends on data quality)

**4. Prediction**
- New image → Extract same 107 features
- Feed to trained model
- Model outputs: disease name + confidence

### Why This Approach Works:
- Diseased leaves have different colors (browns, yellows)
- Different texture patterns (spots, patches)
- Features capture these differences numerically
- ML finds patterns humans might miss

---

## Troubleshooting

### Error: "Model not found"
→ Run `python train_model.py` first

### Error: "Module not found"
→ Install missing library: `pip install <library_name>`

### Low Accuracy (<70%)
→ Use more images per class (increase from 100 to 200)
→ Try different algorithms (SVM instead of Random Forest)

### Supabase connection failed
→ Check `.env` file has correct URL and KEY
→ Verify Supabase project is active

### Images not loading
→ Check image file extensions (.jpg, .jpeg, .png only)
→ Verify dataset folder path is correct

---

## Extending the Project (If You Have More Time)

1. **Add More Plant Types:** Include potato, corn, pepper diseases
2. **Real-time Camera:** Use webcam instead of upload
3. **Mobile App:** Deploy using Streamlit Cloud + mobile browser
4. **Better Features:** Add Local Binary Patterns (LBP) for texture
5. **Deep Learning:** Try transfer learning with MobileNet
6. **Analytics Dashboard:** Show statistics from Supabase data

---

## What You Learned

✅ Image feature extraction (converting images to numbers)  
✅ Supervised ML (Random Forest classification)  
✅ Train-test split and model evaluation  
✅ Web app development (Streamlit)  
✅ Database integration (Supabase)  
✅ End-to-end ML pipeline  

---

## Requirements.txt

```txt
streamlit==1.31.0
scikit-learn==1.4.0
opencv-python==4.9.0.80
Pillow==10.2.0
pandas==2.2.0
numpy==1.26.3
supabase==2.3.4
python-dotenv==1.0.1
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.2
```

---

## Next Steps

1. **Test with various images** - Try healthy vs diseased leaves
2. **Improve accuracy** - Collect more data, try different algorithms
3. **Add features** - Treatment recommendations, pest detection
4. **Deploy online** - Use Streamlit Cloud (free!) to share with others
5. **Add to portfolio** - Great project for resume/GitHub

---

## Questions While Building?

**Q: Why 107 features specifically?**  
A: RGB stats (6) + HSV (3) + texture (2) + histograms (32×3=96) = 107

**Q: Can I use other algorithms?**  
A: Yes! Try SVC, KNN, or even neural networks. Random Forest is a good start.

**Q: Why Random Forest over others?**  
A: Good accuracy, fast training, handles high-dimensional data, less overfitting.

**Q: How to improve accuracy?**  
A: More data, better features (LBP, GLCM), ensemble methods, hyperparameter tuning.

---

**Good luck! You've got this! 🚀**

Any errors? Read error messages carefully - they usually tell you what's wrong!
