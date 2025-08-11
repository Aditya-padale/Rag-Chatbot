# 🚀 Deployment Troubleshooting Guide

## ❌ Common Render Deployment Issues & Solutions

### **Issue 1: Python Version Conflicts**
**Error**: `KeyError: '__version__'` or wheel building errors

**Solution**:
1. ✅ **Use Python 3.11.5** (set in `runtime.txt`)
2. ✅ **Avoid Python 3.13** (too new, causes package issues)

### **Issue 2: Package Version Issues**
**Error**: `subprocess-exited-with-error` during wheel building

**Solutions Applied**:
1. ✅ **Downgraded problematic packages**:
   - `langchain==0.0.352` (stable version)
   - `fastapi==0.103.1` (stable version) 
   - `gunicorn==20.1.0` (stable version)

2. ✅ **Removed problematic dependencies**:
   - Removed `pytesseract` (causes build issues)
   - Simplified image processing

### **Issue 3: Build Script Issues**
**Error**: Build fails during dependency installation

**Solutions Applied**:
1. ✅ **Enhanced build.sh**:
   - Upgrades pip first
   - Installs wheel and setuptools
   - Uses `--no-cache-dir` flag

---

## 🛠️ **Step-by-Step Deployment Fix**

### **1. Render Backend Deployment**

1. **Push these updated files to GitHub**:
   ```bash
   git add .
   git commit -m "Fix deployment issues - stable package versions"
   git push
   ```

2. **In Render Dashboard**:
   - Go to your service
   - **Environment Variables** → Add:
     ```
     GOOGLE_API_KEY = AIzaSyDC9Oa9xK23ZQhP0xkW-Idk3Cn47muh_oY
     GOOGLE_GENERATIVE_AI_API_KEY = AIzaSyDC9Oa9xK23ZQhP0xkW-Idk3Cn47muh_oY
     PYTHON_VERSION = 3.11.5
     ```

3. **Settings**:
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT`
   - **Python Version**: 3.11.5

4. **Deploy**:
   - Click "Manual Deploy" → "Deploy Latest Commit"

### **2. Alternative: Even Simpler Requirements**

If still having issues, try this minimal `requirements.txt`:

```txt
fastapi==0.100.1
uvicorn==0.22.0
python-dotenv==1.0.0
langchain==0.0.300
langchain-google-genai==0.0.11
faiss-cpu==1.7.3
python-docx==0.8.11
pdfplumber==0.7.6
pillow==9.5.0
jinja2==3.1.2
python-multipart==0.0.6
gunicorn==20.1.0
```

### **3. Manual Testing Commands**

Test locally first:
```bash
# Install exact versions
pip install -r requirements.txt

# Test server
uvicorn main:app --host 0.0.0.0 --port 8000

# Test with gunicorn (production server)
gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

---

## ✅ **What We Fixed**

1. **Python Version**: Locked to 3.11.5 (stable)
2. **Package Versions**: Used stable, tested versions
3. **Build Process**: Enhanced with proper pip upgrades
4. **Dependencies**: Removed problematic packages
5. **Runtime**: Added explicit runtime.txt

---

## 🔄 **If Still Having Issues**

### **Option A: Use Docker** (Most Reliable)
Create `Dockerfile`:
```dockerfile
FROM python:3.11.5-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
```

### **Option B: Railway Deployment** (Alternative to Render)
1. Connect GitHub to Railway
2. Add environment variables
3. Automatic deployment

### **Option C: Use Render's Build Logs**
1. Go to Render Dashboard
2. Check "Logs" tab during build
3. Look for specific error messages
4. Address each error individually

---

## 📞 **Support Steps**

1. **Check the current deployment status**
2. **Review Render build logs**
3. **Test locally with exact same versions**
4. **Use the troubleshooting commands above**

The deployment should work now with these fixes! 🎉
