# MorphoPose

📅 Timeline: July 21 – August 15
🔧 Goal: Fully working prototype of MorphPose (species-aware 3D pose estimation with dynamic joints and weak real supervision)

✅ Week-by-Week Overview
Week	Focus	Outcome
Week 1 (July 21–27)	Data Prep + Joint Superset + Synthetic Training	Working synthetic-only MorphPose model
Week 2 (July 28–Aug 3)	Real Data Integration + Weak Supervision Logic	Cross-domain weakly supervised training
Week 3 (Aug 4–10)	Dynamic Masking + Evaluation + Ablations	End-to-end results + analysis
Buffer (Aug 11–15)	Polishing + Visualizations + Hyperparams	Ready for writing/paper submission

🗓️ WEEK 1: Synthetic Training and Core Model
📍July 21–22: Setup
✅ Organize data

Extract joint presence masks from synthetic 3D

Define J_superset (global joint list)

✅ Preprocess synthetic 2D and 3D into training format (with presence mask)

✅ Implement dataloader with masking logic

📍July 23–24: MorphPose Architecture
✅ Build base model: takes 2D seq → predicts:

3D joints in superset

Joint presence mask

✅ Choose temporal backbone (TemporalConv1D is fast)

✅ Train on synthetic only (supervised loss)

📍July 25–27: Synthetic Evaluation
✅ Evaluate on held-out synthetic species

✅ Visualize 3D skeleton + joints

✅ Ablate dynamic joint mask vs fixed joint version

🗓️ WEEK 2: Real Data & Weak Supervision
📍July 28–29: Real Data Integration
✅ Run pretrained 2D pose detector on real images/videos

✅ Track across time (if possible)

✅ Store pseudo-2D keypoints

📍July 30–31: Joint Mapping + Loader
✅ Define mapping A from real → superset joints

✅ Build real data loader with mapped pseudo-2D

✅ Integrate into training loop

📍Aug 1–3: Weak Supervised Real Training
✅ Add weak reprojection loss using mapped joints:

L_2D_real = || P_reproj[real_joints] - pseudo_2D ||

✅ Mix real + synthetic batches (50-50)

✅ Add optional regularizations:

Temporal smoothness

Pose prior or latent space constraint (optional)

🗓️ WEEK 3: Dynamic Masking, Evaluation, Ablations
📍Aug 4–5: Evaluate on Real Dogs
✅ Visualize real 3D output (MeshCat / Open3D / matplotlib)

✅ Compare reprojection errors

✅ Show generalization to real breeds not seen in synthetic

📍Aug 6–7: Advanced (Optional)
✅ Add latent embedding for species

✅ Add confidence-aware joint mask (uncertainty modeling)

📍Aug 8–10: Ablations + Analysis
✅ Remove dynamic mask → fixed skeleton baseline

✅ Remove synthetic → see performance drop

✅ Track metrics: reprojection error, joint activation, latent consistency

📅 Aug 11–15: BUFFER + VISUALIZATION
✅ Render qualitative results: real dog → 3D pose with varying skeletons

✅ Save animations or videos

✅ Final tuning of training schedules / LR / loss weights

✅ Back up models, training logs, and plots
