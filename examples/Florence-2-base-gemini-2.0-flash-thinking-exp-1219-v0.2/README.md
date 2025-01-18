---
datasets:
- >-
  PJMixers-Images/Handpicked-Images-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT
- >-
  PJMixers-Images/bghira_pseudo-camera-10k-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT
- >-
  PJMixers-Images/r_analog-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT
- >-
  PJMixers-Images/r_portraitphotography-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT
language:
- en
base_model:
- microsoft/Florence-2-base
pipeline_tag: image-to-text
library_name: transformers
---
# [Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2)

A [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) finetune intended to improve the quality and formatting of image captioning.

This is a very early version as I have a limited number of training images and the variety of images is lacking. Basically all images used were *high quality* so captioning *low quality* images may perform poorly.

## Model Evaluation

| Source Image | Ground Truth | Original Model Prediction | Fine-tuned Model Prediction |
|--------------|--------------|---------------------------|-----------------------------|
| ![eval_img_1](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_1.jpg) | A woman with fair skin and dark brown hair styled with short bangs and an updo looks directly at the viewer, holding a fluffy grey cat in her arms. The woman has brown eyes, visible pink blush on her cheeks, and dark pink lipstick. She wears small silver hoop earrings and a white long-sleeved blouse with a high, ruffled collar and ruffled cuffs. The cat has long, grey fur and green eyes, and it is looking to the left of the frame. The background is a solid purple color with a slightly textured appearance. The woman's left hand is holding the cat's chest, and her right hand is supporting the cat's rear. The bottom portion of the woman's garment appears to be a tan or light brown color. | A woman holding a cat in her arms. | A woman with fair skin and short brown hair is holding a fluffy gray and white cat in her arms. She is wearing a white long-sleeved shirt with a ruffled collar and white lace detailing on the sleeves. Her eyes are blue, and she has dark eyeliner and lipstick. She has silver hoop earrings in her left ear. The cat has green eyes and is looking directly at the viewer. The background is a solid purple color. |
| ![eval_img_2](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_2.png) | A vibrant green leaf, positioned centrally in the frame, is covered in numerous glistening water droplets. The leaf has a pointed tip and smooth edges, with prominent veins running vertically along its surface. The water droplets vary in size, some appearing larger and more defined, while others are smaller and more scattered. The background is a solid, uniform green color, providing a stark contrast to the leaf and emphasizing its details. The lighting creates highlights and shadows on the leaf, particularly noticeable on the water droplets, adding depth and dimension to the image. | A green leaf with water droplets on it. | A close-up view shows a vibrant green leaf with numerous water droplets scattered across its surface, positioned in the center of the frame against a solid green background. |
| ![eval_img_3](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_3.png) | A smiling woman with blonde hair is positioned on the left side of the image, wearing a beige knitted sweater with long, slightly puffed sleeves and white pants. Her right hand is raised near her right temple, with a silver ring visible on her finger, and she has a silver hoop nose piercing and a black stud earring in her left ear. A green leafy plant is situated on the right side of the image. The background consists of a plain white wall, with a shadow cast by the woman on the left side and a shadow cast by the plant on the right side. | A woman wearing a beige sweater and white pants. | A fair-skinned woman with blonde hair is positioned in the center of the frame against a plain white wall. She is wearing a long-sleeved, light brown, ribbed knit sweater that reaches her shoulders. Her right hand is raised above her head, with her index finger touching her hair. She has a silver hoop earring in her left ear. Her eyes are closed and she has a slight smile on her lips. Her hair is parted in the middle, and she is wearing small silver stud earrings. The background is a solid white color. To the right of the woman is a green leafy plant with large, oval-shaped leaves. |
| ![eval_img_4](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_4.png) | A body of water fills the foreground, reflecting the sky and the trees on the opposite bank. The reflection of a large tree in the center is clearly visible in the water, mirroring its shape and the green of its leaves. Patches of brown and tan vegetation surround the edges of the water, with some sparse green reeds growing within the water itself. A large tree stands prominently on the right side of the water's edge, its branches extending outwards and its green foliage dense. To the left of this central tree, two other clusters of trees are visible, their foliage also green. In the background, a line of trees forms a horizon, their darker green contrasting with the lighter vegetation in the midground. The sky above is light blue with scattered white clouds. | A small pond in the middle of a grassy field. | A body of water fills the lower portion of the image, reflecting the surrounding vegetation and the sky above. The water is a dark blue-grey color, and the reflections of the trees and vegetation are clearly visible on its surface. On the right side of the water, there is a patch of dry, golden-brown grass that is partially submerged in the water. Behind the grass, there are more dry, dry grasses. In the background, a line of trees with green and brown leaves is visible. The sky above is light blue with scattered white clouds. |
| ![eval_img_5](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_5.png) | Two women are lying down next to each other, with the woman on the left having her eyes closed and smiling, revealing her white teeth, and wearing a light-colored floral shirt with blue, pink, and white flowers, and a gold necklace with the legible text "ABENA" on it, and her dark hair styled in twists, with her left hand raised near the other woman's head, showing a gold ring on her finger and a gold bracelet on her wrist, and her fingernails painted a light color. The woman on the right has her eyes open, revealing her light-colored irises, and is smiling, showing her white teeth, and wearing a pink, blue, and black patterned top with a gold necklace, and her dark hair is also styled in twists. They are lying on a dark purple surface. | A woman and a young girl laying on top of each other. | Two women are lying on top of each other, with the woman on the left having her eyes closed and her mouth open, showing her teeth, and she has dark hair styled in braids. She is wearing a light blue collared shirt with a floral pattern and a gold necklace with the word "ABENA" printed on it. Her right hand is raised with her fingers spread apart, wearing a gold ring on her left ring finger, and her fingernails are painted with white nail polish. Her eyes are closed and she is wearing gold hoop earrings. Her mouth is open, revealing her teeth. She has dark blue eyes and is looking directly at the viewer with a slight smile. Her black dreadlocks are visible on the right side of the image. The top of the top has a colorful pattern of black, pink, blue, yellow, orange, and white circles. There is also a gold decorative element on the top right corner. The background is a dark purple surface. |
| ![eval_img_6](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_6.png) | A low angle view shows the upper portion of an ornate, light brown stone building against a clear blue sky with a few wispy white clouds at the top. The building features intricate carvings and patterns across its facade, including geometric designs and floral motifs. A triangular pediment tops the structure, adorned with a central circular design and smaller decorative elements. Below the pediment is a horizontal band with shell-like carvings. A rectangular sign is visible on the facade, containing text. Two rounded finials sit atop the corners of the pediment. Two cylindrical columns are visible on the left side of the frame, and a portion of another column is seen on the right. A black metal gate with vertical bars is partially visible in the lower center of the image, obscuring the entrance to the building. The stone of the building appears weathered and textured. | A large stone building with a gate in front of it. | A low angle view shows the facade of a building under a bright blue sky with scattered white clouds. The building is made of light-colored stone and features intricate carvings and decorative elements. At the top of the building, there is a decorative pediment with a round finial at the top. Below the pediment, the word "Wizard" is visible in blue capital letters. To the left of this pediment is a section of a red brick wall with vertical columns. The central section of the facade features a carved geometric pattern. Below this central section is an open doorway with a black metal gate with vertical bars. The gate is framed by a decorative border. The brick wall also has several arched windows with visible panes. |
| ![eval_img_7](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_7.png) | A man with short brown hair and wearing sunglasses and an olive green jacket is leaning against the hood of a dark-colored car, facing a woman with curly brown hair wearing sunglasses and a blue denim jacket and grey pants. The man has his left arm around the woman's back. They are standing on a paved area with a road visible behind them. Beyond the road is a grassy area with green trees and a tall metal electricity pylon. The sky is cloudy with patches of blue. | A man and a woman standing next to a car. | A man with short brown hair and sunglasses is sitting on the hood of a grey car, wearing a green long-sleeved shirt and blue jeans, with his left arm outstretched towards a woman with curly brown hair wearing a blue denim jacket and black sunglasses. The woman is standing next to the man, with her right arm around the man's waist. The man is looking to the left. The car is parked on a paved road with a white line marking the edge. Behind the road is a grassy area with green and yellow flowers. In the background, there are trees and a power line tower. The sky is light blue with scattered white clouds. |
| ![eval_img_8](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/eval_img_8.png) | A fair-skinned woman with short blonde curly hair and glasses is reaching up with both hands to adjust a decorative wreath hanging on a dark teal doorframe. She is wearing a red three-quarter sleeve shirt over a white lace top, and has a gold bracelet on her left wrist and a ring on her right hand. The wreath is made of brown and orange artificial leaves and twigs, and is attached to the doorframe with an orange ribbon. The doorframe has a decorative black wrought iron arch with circular details. Sunlight is casting shadows of leaves and branches on the right side of the doorframe and the wall behind it. The background shows a glimpse of a light-colored wall with a patterned design. | A woman putting leaves on a wreath on a door. | A woman with short blonde hair and glasses, wearing a red short-sleeved shirt and a white lace top, is looking to the left and to the right, with her right hand holding a wreath made of autumn leaves. The wreath is attached to a black metal gate with a curved top. The leaves are a mix of brown and orange, with some appearing darker and others lighter. The woman is wearing a gold bracelet on her left wrist and a ring on her right ring finger. The gate has a dark metal frame with a decorative finial. The background is blurred, but it appears to be an outdoor setting with a light-colored wall and a dark-colored object on the left side. |

![val_loss](https://huggingface.co/PJMixers-Images/Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2/resolve/main/images/val_loss.png)

## Training Settings

Trained with [Florence-2ner](https://github.com/xzuyn/Florence-2ner) using this config and ~4.5K images from these datasets:

- [PJMixers-Images/Handpicked-Images-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT](https://huggingface.co/datasets/PJMixers-Images/Handpicked-Images-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT/blob/3a73b6b910b87d72c23e6a57a3a4f4a0e0564008/images.zip)
- [PJMixers-Images/bghira_pseudo-camera-10k-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT](https://huggingface.co/datasets/PJMixers-Images/bghira_pseudo-camera-10k-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT/blob/e240ee8e7f19212fc26d22bb9da572dd801b4e8a/images.zip)
- [PJMixers-Images/r_analog-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT](https://huggingface.co/datasets/PJMixers-Images/r_analog-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT/blob/0d1ab6a5cfe5900806ded347d29fd4d0ae337320/images.zip)
- [PJMixers-Images/r_portraitphotography-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT](https://huggingface.co/datasets/PJMixers-Images/r_portraitphotography-gemini-2.0-flash-thinking-exp-1219-CustomShareGPT/blob/e6640ad80debe7b139b71aa230199a15ca8c5683/images.zip)

```json
{
    "model_name": "microsoft/Florence-2-base",
    "task_prompt": "<CAPTION>",
    "dataset_path": "./0000_Datasets/Gemini",
    "wandb_project_name": "Florence-2-base",
    "run_name": "Florence-2-base-gemini-2.0-flash-thinking-exp-1219-v0.2-run1",
    "epochs": 2,
    "optimizer": "CAME",
    "learning_rate": 3e-6,
    "lr_scheduler": "REX",
    "gradient_checkpointing": true,
    "freeze_vision": false,
    "freeze_language": false,
    "freeze_other": false,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "clip_grad_norm": 1,
    "weight_decay": 1e-2,
    "save_total_limit": 3,
    "save_steps": 10,
    "eval_steps": 10,
    "warmup_steps": 10,
    "eval_split_ratio": 0.05,
    "seed": 42,
    "filtering_processes": 128,
    "attn_implementation": "sdpa"
}
```

