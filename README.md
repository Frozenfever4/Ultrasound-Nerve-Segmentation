# Ultrasound-Nerve-Segmentation
Ultrasound Nerve Segmentation Using Diffusion-Based Denoising and Attention-Driven Feature Fusion


# Abstract: 
Segmentation in ultrasound images is difficult in the presence of speckle noise, poor contrast, and ambiguous anatomical edges. This paper presents a multi-channel segmentation approach that combines raw ultrasound image data, denoised features, and maps highlighting anatomical edges for enhanced feature representation. A new EdgeFusion-Unet model has been proposed that enables accurate fusion of all the above features for effective boundary detection. The proposed model has been compared in terms of the Dice score, precision, and F1-measure to the UNet, Residual UNet, and attention UNet models. Experimental analysis clearly indicates that the proposed approach has resulted in more accurate segmentation and has retained edges effectively compared to the existing approaches.

# Problem Statement:
Identifying nerves in ultrasound images is inherently challenging due to speckle noise, low contrast, and poorly defined tissue boundaries that obscure fine anatomical details. Despite these challenges, accurate nerve segmentation is critical for clinical procedures such as regional anesthesia, pain-management catheterization, and trauma care, where precise nerve localization directly impacts procedural safety and success. Conventional post-operative pain management often relies on narcotic analgesics, which are associated with adverse effects including nausea, sedation, addiction risk, and prolonged recovery. Consequently, there is growing clinical interest in catheter-based nerve blocks that deliver localized pain relief; however, their effectiveness depends on reliable nerve identification during ultrasound guidance. This has driven the need for automated and accurate nerve segmentation systems, as emphasized by the Kaggle ultrasound nerve segmentation challenge.

Beyond anesthesia, nerve segmentation plays an important role in trauma medicine. While fractures are readily diagnosed, nerve injuries are difficult to detect in early stages and may lead to long-term complications if missed. Neck nerves, which are less prone to crushing injuries than peripheral nerves, offer a consistent anatomical region for ultrasound-based modeling. An effective nerve segmentation system can therefore support rapid nerve detection and integrity assessment in emergency settings, enabling timely and informed clinical interventions.


