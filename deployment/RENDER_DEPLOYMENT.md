# Deployment on Render

## Steps to Deploy on Render

### 1. Prepare Your Repository
- Commit all changes to your branch:
```bash
git add .
git commit -m "Setup Render deployment files"
git push origin your-branch-name
```

### 2. Create a Render Account
- Go to https://www.render.com
- Sign up or log in with your GitHub account

### 3. Create a New Web Service
1. Go to Dashboard > New +
2. Select "Web Service"
3. Connect your GitHub repository
4. Select the appropriate repository

### 4. Configure the Service
- **Name**: energy-prediction-api (or your preferred name)
- **Branch**: your-branch-name
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python backend/app.py`

### 5. Set Environment Variables (if needed)
- Under "Environment", add any custom variables
- The PORT environment variable is automatically provided by Render

### 6. Deploy
- Click "Create Web Service"
- Render will automatically deploy your application
- Once deployed, you'll get a URL like `https://your-app.onrender.com`

### 7. Update Frontend URL
The frontend will automatically detect if it's running locally or deployed and adjust the backend URL accordingly.

### Important Notes
- Ensure `requirements.txt` is in the root directory ✓
- Ensure `Procfile` is in the root directory ✓
- Ensure `runtime.txt` specifies the Python version ✓
- The model file `models/energy_lstm_model.h5` should be tracked in GitHub (use Git LFS if file is too large)

## Troubleshooting

### Model Not Found Error
If you see "Model load error", make sure:
1. The `models/energy_lstm_model.h5` file is committed to Git
2. The file path in `backend/app.py` is correct

### Port Issues
- Render automatically assigns a PORT environment variable
- The app is configured to use this PORT automatically

### CORS Issues
- Flask-CORS is already configured in app.py
- All requests should work after deployment

## Testing Your Deployment
Once deployed, you can:
1. Test the backend API directly: `https://your-app.onrender.com/`
2. Deploy the frontend separately or serve it from the same domain
3. The backend will handle CORS requests from any origin
