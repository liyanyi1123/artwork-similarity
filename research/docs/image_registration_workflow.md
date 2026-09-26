# Image Registration Experiment Workflow

```mermaid
flowchart LR
    START([Start])

    subgraph S1[Stage 1: Data Preparation]
        direction TB
        A1[Load 2,000 dataset images]
        A2[Extract features with CLIP]
        A3[Extract features with ViT]
        A4[(Vector database: 2,000 CLIP / ViT embedding pairs and artist DIDs)]
        A1 --> A2
        A1 --> A3
        A2 --> A4
        A3 --> A4
    end

    subgraph S2[Stage 2: Query Image Processing]
        direction TB
        B1[/User uploads one image and artist DID/]
        B2[Extract the query image's CLIP and ViT embeddings]
        B3[Compute CLIP and ViT similarities against all 2,000 database images]
        B4[Take the absolute value of every similarity score]
        B1 --> B2 --> B3 --> B4
    end

    subgraph S3[Stage 3: Threshold Check and Decision]
        direction TB
        C1{Do all 2,000 database images satisfy:<br/>Absolute CLIP similarity &lt; 0.87<br/>OR Absolute ViT similarity &lt; 0.50?}
        C2{Do the query image and matched image belong to the same artist?<br/>Compare artist DIDs}
        C1 -->|No: Similar artwork found| C2
    end

    subgraph S4[Stage 4: Registration Result]
        direction TB
        D1[Display: Registration successful]
        D2[Display: Registration failed]
        D3([End])
        D1 --> D3
        D2 --> D3
    end

    START --> A1
    A4 --> B1
    B4 --> C1
    C1 -->|Yes: No similar artwork found| D1
    C2 -->|Yes: Same artist| D1
    C2 -->|No: Different artist| D2
```

Equivalently, no database image may simultaneously satisfy `|CLIP similarity| >= 0.87` and `|ViT similarity| >= 0.50`.

If a similar artwork is found, compare the artist DIDs: registration succeeds for the same artist and fails for a different artist.
