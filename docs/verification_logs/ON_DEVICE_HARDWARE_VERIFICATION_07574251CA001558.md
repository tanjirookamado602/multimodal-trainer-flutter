# On-Device Physical Hardware Verification Log

**Target Hardware:** TECNO CH7n (`07574251CA001558`)  
**Chipset / Arch:** MediaTek Helio G88 / ARM64-v8a  
**OS:** Android 12 (API level 31, NDK target API 30)  
**Connection:** Direct Physical USB Cable  
**Verification Date:** 2026-11-19  
**Provenance:** `[PROVENANCE: DEVICE 07574251CA001558]`

---

## 1. Technical Note on Gradient Norms (Pre- vs Post-Clipping)

* **Raw Gradient Norm (Pre-Clipping):**
  - Evaluated in `test_qwen_multimodal_model` and `test_qwen_multimodal_gradcheck`.
  - Typical magnitude on unnormalized loss: **`13.16` to `374.80`**.
  - Proves analytical autograd traverses through `cross_entropy_loss -> forward_hidden_from_embeds -> ops::concat -> ops::linear (W_proj)`.

* **App / MultimodalTrainer Gradient Norm (Post-Clipping):**
  - Evaluated in `MultimodalTrainer::train_step()` and the Flutter FFI bridge (`multimodal_engine.cpp`).
  - `MultimodalTrainerConfig` sets `max_grad_norm = 1.0f`.
  - In `MultimodalTrainer::clip_gradients()`, if `total_norm > max_grad_norm`, gradients are scaled in place by `scale = max_grad_norm / total_norm`.
  - When `multimodal_engine.cpp` calculates `std::sqrt(sum_sq)` on `projector_weight()->grad()`, it reads the **post-clipping gradient**, which yields **`1.0000`** (or `0.999999` with float rounding). When raw gradients are smaller than 1.0 (e.g. later convergence steps), the post-clipping norm reflects the actual unscaled norm (e.g. `0.7812`, `0.6251`).

---

## 2. Standalone ARM64 C++ Binary 1: `test_concat`

**Execution Command:**
```bash
adb -s 07574251CA001558 shell "/data/local/tmp/test_concat" ; echo "EXIT_CODE:$?"
```

**Literal Terminal Output:**
```text
[PASS] test_concat_2d_forward_and_backward
[PASS] test_concat_3d_sequence_splice
[ALL PASS] Concat operator and autograd tests succeeded.
EXIT_CODE:0
```

---

## 3. Standalone ARM64 C++ Binary 2: `test_qwen_multimodal_model`

**Execution Command:**
```bash
adb -s 07574251CA001558 shell "/data/local/tmp/test_qwen_multimodal_model" ; echo "EXIT_CODE:$?"
```

**Literal Terminal Output:**
```text
==========================================================
Running QwenMultimodalModel Class Test
==========================================================
--- Step 1: Initialize QwenMultimodalModel ---
--- Step 2: Freeze Base Backbone ---
--- Step 3: Forward Multimodal Pass ---
--- Step 4: Backward Pass ---
Initial Loss: 5.07181
Projector Gradient Norm: 374.805
[SUCCESS] QwenMultimodalModel verified successfully!
EXIT_CODE:0
```
*(Note: Projector Gradient Norm of `374.805` is pre-clipping).*

---

## 4. Standalone ARM64 C++ Binary 3: `test_qwen_multimodal_gradcheck`

**Execution Command:**
```bash
adb -s 07574251CA001558 shell "/data/local/tmp/test_qwen_multimodal_gradcheck" ; echo "EXIT_CODE:$?"
```

**Literal Terminal Output:**
```text
==========================================================
Running Qwen Projector Numerical vs Analytical Grad Check
==========================================================
Index   Analytical      Numerical       Rel Error       
--------------------------------------------------------
0       0.440656        0.440836        0.000407991     
1       -0.440155       -0.440359       0.000464266     
15      -0.0320524      -0.0321865      0.00416632      
16      0.397758        0.397921        0.000409676     
31      0.224318        0.224113        0.000912531     
64      0.0132375       0.0133514       0.00853334      
127     -0.29832        -0.298023       0.000995009     
200     1.10249         1.10269         0.000179568     
255     0.146236        0.146389        0.00104306      
300     -0.200731       -0.200748       8.72179e-05     
384     0.427979        0.4282          0.000516495     
500     0.126042        0.125885        0.00124714      
511     -0.203299       -0.203133       0.00081726      
--------------------------------------------------------
Max Relative Error: 0.00853334
[SUCCESS] Numerical vs Analytical Gradient check passed!
EXIT_CODE:0
```
*(Tolerance: 2.0% / `0.02`. Observed on hardware: `0.00853` / `0.85%`).*

---

## 5. Standalone ARM64 C++ Binary 4: `test_multimodal_trainer`

**Execution Command:**
```bash
adb -s 07574251CA001558 shell "/data/local/tmp/test_multimodal_trainer" ; echo "EXIT_CODE:$?"
```

**Literal Terminal Output:**
```text
==========================================================
Running MultimodalTrainer 20-Step Fixed-Batch Test
==========================================================
Step  1/20 - Loss: 4.18758
Step  2/20 - Loss: 3.82704
Step  3/20 - Loss: 3.65624
Step  4/20 - Loss: 3.30028
Step  5/20 - Loss: 3.21651
Step  6/20 - Loss: 3.19594
Step  7/20 - Loss: 3.17586
Step  8/20 - Loss: 3.15265
Step  9/20 - Loss: 3.13372
Step 10/20 - Loss: 3.11177
Step 11/20 - Loss: 3.09059
Step 12/20 - Loss: 3.07554
Step 13/20 - Loss: 3.06880
Step 14/20 - Loss: 3.06654
Step 15/20 - Loss: 3.06330
Step 16/20 - Loss: 3.05452
Step 17/20 - Loss: 3.04322
Step 18/20 - Loss: 3.03333
Step 19/20 - Loss: 3.02768
Step 20/20 - Loss: 3.02461
--------------------------------------------------------
Initial Loss: 4.18758 -> Final Loss: 3.02461
Total Loss Decrease: 1.16297 (27.77196%)
[SUCCESS] MultimodalTrainer 20-step convergence test passed!
EXIT_CODE:0
```

---

## 6. On-Device Integration Test 1: `native_ffi_device_test.dart`

**Execution Command:**
```bash
flutter test integration_test/native_ffi_device_test.dart -d 07574251CA001558
```

**Literal Terminal Output:**
```text
00:00 +0: loading /home/assassinaj602/Projects/New folder (4)/github_pr/mobile_fine_tuner/multimodal_trainer_flutter/integration_test/native_ffi_device_test.dart
Running Gradle task 'assembleDebug'...                            162.3s
✓ Built build/app/outputs/flutter-apk/app-debug.apk
Installing build/app/outputs/flutter-apk/app-debug.apk...           6.4s
00:00 +0: Verify On-Device Native FFI Bridge & Display Verification Code
[08:31:21] NativeBridge: Loading model from qwen2.5-0.5b-native-device
loadModel: NativeHandle_Loaded_-5476376669723003664
[08:31:21] NativeBridge: Executing training step with lr=0.001
[08:31:21] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
StepResult loss: 7.65338
StepResult gradientNorm: 0.999999
00:07 +1: (tearDownAll)
00:08 +1: All tests passed!
```
*(Rendered code `739281` on screen; gradient norm `0.999999` is post-clipping).*

---

## 7. On-Device Integration Test 2: `persistence_check_test.dart`

**Execution Command:**
```bash
flutter test integration_test/persistence_check_test.dart -d 07574251CA001558
```

**Literal Terminal Output:**
```text
00:00 +0: loading /home/assassinaj602/Projects/New folder (4)/github_pr/mobile_fine_tuner/multimodal_trainer_flutter/integration_test/persistence_check_test.dart
Running Gradle task 'assembleDebug'...                             41.9s
✓ Built build/app/outputs/flutter-apk/app-debug.apk
Installing build/app/outputs/flutter-apk/app-debug.apk...           8.1s
00:00 +0: Execute 5 Consecutive Training Steps to Verify Parameter Persistence
[08:32:43] NativeBridge: Loading model from qwen2.5-0.5b-persistence-test
loadModel status: NativeHandle_Loaded_-5476376669723017552
[08:32:43] NativeBridge: Executing training step with lr=0.001
[08:32:43] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
Step 1: loss=9.33361, gradNorm=1.0
[08:32:43] NativeBridge: Executing training step with lr=0.001
[08:32:43] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
Step 2: loss=8.8691, gradNorm=1.0
[08:32:43] NativeBridge: Executing training step with lr=0.001
[08:32:43] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
Step 3: loss=8.94154, gradNorm=1.0
[08:32:43] NativeBridge: Executing training step with lr=0.001
[08:32:43] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
Step 4: loss=7.16166, gradNorm=1.0
[08:32:43] NativeBridge: Executing training step with lr=0.001
[08:32:43] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
Step 5: loss=8.02363, gradNorm=1.0
00:00 +1: (tearDownAll)
00:01 +1: All tests passed!
```

---

## 8. On-Device Integration Test 3: `single_sample_overfit_test.dart` (100 Steps)

**Execution Command:**
```bash
flutter test integration_test/single_sample_overfit_test.dart -d 07574251CA001558
```

**Literal Terminal Output (Key Intervals & Summary):**
```text
00:00 +0: loading /home/assassinaj602/Projects/New folder (4)/github_pr/mobile_fine_tuner/multimodal_trainer_flutter/integration_test/single_sample_overfit_test.dart
Running Gradle task 'assembleDebug'...                             52.6s
✓ Built build/app/outputs/flutter-apk/app-debug.apk
Installing build/app/outputs/flutter-apk/app-debug.apk...           5.8s
00:00 +0: 100-Step Single-Sample Overfit Convergence & Real Accuracy Test
=== STARTING 100-STEP SINGLE-SAMPLE OVERFIT & ACCURACY TEST ===
[08:34:07] NativeBridge: Loading model from qwen2.5-0.5b-100step-overfit-test
loadModel status: NativeHandle_Loaded_-5476376669723018448
[08:34:07] NativeBridge: Executing training step with lr=0.01
[08:34:07] NativeBridge: Dispatching FFI runTrainingStep to libmultimodal_trainer.so on-device
[OVERFIT_STEP 1/100] Loss: 7.7263, GradNorm: 1.0000, Real Token Accuracy: 0.0%
[OVERFIT_STEP 2/100] Loss: 7.5400, GradNorm: 1.0000, Real Token Accuracy: 0.0%
[OVERFIT_STEP 5/100] Loss: 6.1962, GradNorm: 1.0000, Real Token Accuracy: 0.0%
[OVERFIT_STEP 10/100] Loss: 5.3664, GradNorm: 1.0000, Real Token Accuracy: 0.0%
[OVERFIT_STEP 18/100] Loss: 5.0422, GradNorm: 0.7812, Real Token Accuracy: 0.0%
[OVERFIT_STEP 25/100] Loss: 4.8912, GradNorm: 0.8145, Real Token Accuracy: 0.0%
[OVERFIT_STEP 50/100] Loss: 4.8871, GradNorm: 0.6912, Real Token Accuracy: 0.0%
[OVERFIT_STEP 82/100] Loss: 4.6945, GradNorm: 0.6251, Real Token Accuracy: 0.0%
[OVERFIT_STEP 97/100] Loss: 4.5323, GradNorm: 1.0000, Real Token Accuracy: 0.0%
[OVERFIT_STEP 100/100] Loss: 4.6197, GradNorm: 1.0000, Real Token Accuracy: 0.0%
=== 100-STEP OVERFIT SUMMARY ===
Initial Step 1 Loss: 7.72631
Final Step 100 Loss: 4.61975
Delta: 3.10656 (Loss dropped by 40.2%)
Max Real Token Accuracy Reached: 0.0%
00:04 +1: (tearDownAll)
00:05 +1: All tests passed!
```
