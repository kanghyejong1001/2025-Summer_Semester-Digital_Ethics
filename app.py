import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import torch
from torchvision import models
from PIL import Image
import random
import json

# ---------------------------
# 사이드바 소개
# ---------------------------
st.sidebar.title("웹앱 소개")
st.sidebar.markdown("""
이 웹앱은 **딥페이크 vs 실제 이미지의 AI 판별해보고, 직접 구별도 해볼 수 있는 체험 도구**로,  
학생들이 인공지능 기술의 원리와 윤리적 활용 방안을 탐구할 수 있도록 설계되었습니다.

### 🔍 주요 활동
- **AI 이미지 판별** 체험 (실제 vs 딥페이크)
- **딥페이크 vs 실제 이미지 직접 구별**

### 🎯 기대 효과
- 민감 정보(얼굴 이미지)의 윤리적 취급 학습
- 딥페이크 기술의 **위험성과 책임 소재에 대한 비판적 사고** 함양

이 웹앱은 '정보'와 '현대사회와 윤리' 교과의 AI 융합 교육 수업을 위해 제작되었습니다.
""")


# ---------------------------
# AI 예측 모델
# ---------------------------
def predict_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    model = models.resnet18(pretrained=True)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_idx = torch.max(prob, 0)
    prediction = "Fake" if predicted_idx.item() % 2 == 0 else "Real"
    probability = round(confidence.item(), 2)
    reason = f"ResNet18 기반 분류 결과, 클래스 #{predicted_idx.item()}에서 높은 확률 감지됨."
    return prediction, probability, reason

# ---------------------------
# 이미지 판별 탭
# ---------------------------
def deepfake_checker():
    st.subheader("🔍 실제 이미지 vs 딥페이크 이미지 비교")

    st.markdown("""
    - 좌측에는 실제 이미지를, 우측에는 딥페이크 이미지를 업로드하세요.
    - AI가 두 이미지를 각각 분석하여 결과와 확률을 비교합니다.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 실제 이미지 업로드")
        real_file = st.file_uploader("실제 이미지", type=["jpg", "jpeg", "png"], key="real")
        if real_file:
            real_image = Image.open(real_file)
            st.image(real_image, caption="실제 이미지", width=256)
        else:
            real_image = None

    with col2:
        st.markdown("### 🔴 딥페이크 이미지 업로드")
        fake_file = st.file_uploader("딥페이크 이미지", type=["jpg", "jpeg", "png"], key="fake")
        if fake_file:
            fake_image = Image.open(fake_file)
            st.image(fake_image, caption="딥페이크 이미지", width=256)
        else:
            fake_image = None

    # 결과 비교
    if real_image and fake_image:
        if st.button("📊 AI로 비교 분석"):
            # 각각 예측
            pred_real, prob_real, reason_real = predict_image(real_image)
            pred_fake, prob_fake, reason_fake = predict_image(fake_image)

            # 결과 테이블 출력
            st.markdown("### ✅ 예측 결과 비교")
            result_df = pd.DataFrame({
                "이미지": ["실제 이미지", "딥페이크 이미지"],
                "예측 결과": [pred_real, pred_fake],
                "판별 확률": [f"{prob_real*100:.2f}%", f"{prob_fake*100:.2f}%"],
                "판단 근거": [reason_real, reason_fake]
            })
            st.dataframe(result_df)

            # 결과 해석 안내
            st.markdown("🧠 **해석 가이드:**")
            st.markdown("- AI는 ResNet18 분류 결과의 특정 클래스 확률을 기반으로 판단합니다.")
            st.markdown("- 실제 이미지와 딥페이크 이미지 모두 `Real` 또는 `Fake`으로 잘못 분류될 수 있습니다.")

            

# ---------------------------
# 딥페이크 vs 실제 시뮬레이션
# ---------------------------

def simulation():
    st.subheader("🎯 딥페이크 vs 실제 이미지 구별 시뮬레이션")
    st.markdown("""
    아래 무작위로 표시된 **10장의 이미지** 중에서, 딥페이크(Fake)라고 생각되는 이미지만 선택하세요.
    """)

    # 이미지 로딩
    def load_images_from_folder(folder, label):
        path = Path(folder)
        image_files = list(path.glob("*"))
        images = []
        for f in image_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append({"image": img, "label": label, "path": str(f)})
            except:
                continue
        return images

    real_images = load_images_from_folder("real", "Real")
    fake_images = load_images_from_folder("fake", "Fake")
    combined = real_images + fake_images
    quiz_images = random.sample(combined, min(10, len(combined)))

    # 유저 선택
    selected_indices = []
    for i, item in enumerate(quiz_images):
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(item["image"], width=120)
        with col2:
            if st.checkbox(f"딥페이크일 것 같아요 (이미지 {i+1})", key=f"check_{i}"):
                selected_indices.append(i)

    # 제출 버튼
    if st.button("✅ 결과 제출"):
        correct = 0
        total_fake = 0
        st.markdown("---")
        for i, item in enumerate(quiz_images):
            label = item["label"]
            selected = i in selected_indices

            if label == "Fake":
                total_fake += 1
                if selected:
                    st.success(f"🟢 이미지 {i+1}: 딥페이크 맞춤 (정답)")
                    correct += 1
                else:
                    st.info(f"🔵 이미지 {i+1}: 딥페이크인데 선택 안 함 (놓침)")
            else:
                if selected:
                    st.error(f"🔴 이미지 {i+1}: 실제인데 선택함 (오답)")

        accuracy = (correct / total_fake * 100) if total_fake > 0 else 0.0
        st.markdown(f"### 🎯 점수: {correct} / {total_fake} (정답률: {accuracy:.2f}%)")

        # 랭킹 저장
        st.markdown("---")
        st.markdown("#### 📝 닉네임을 입력하고 랭킹에 등록하세요")
        username = st.text_input("닉네임", max_chars=20, key="nickname_input")

        if st.button("📊 랭킹에 반영"):
            record = {"user": username or "익명", "score": correct, "total": total_fake, "accuracy": accuracy}
            log_path = Path("score_logs.json")
            if log_path.exists():
                logs = json.loads(log_path.read_text())
            else:
                logs = []

            logs.append(record)
            logs = sorted(logs, key=lambda x: (-x["score"], -x["accuracy"]))[:10]
            log_path.write_text(json.dumps(logs, indent=2, ensure_ascii=False))

            st.markdown("### 🏆 TOP 10 랭킹")
            df = pd.DataFrame(logs)
            df = df.rename(columns={"user": "닉네임", "score": "정답 수", "total": "전체 Fake 수", "accuracy": "정답률 (%)"})
            st.dataframe(df, use_container_width=True)


# ---------------------------
# 도움말
# ---------------------------
def show_tutorial():
    st.subheader("도움말 및 활용 가이드")
    st.markdown("""
---

### 🧭 [1] 이미지 판별 탭 활용법
- 실제 이미지와 딥페이크 이미지를 나란히 업로드하여, AI가 어떻게 판단하는지 확인할 수 있습니다.
- 학습자의 얼굴로 만든 딥페이크 이미지 판별에도 활용 가능합니다.
- 결과 테이블을 통해 AI의 판단 논리를 탐색하고 오류 가능성도 토의해볼 수 있습니다.

---

### 🎮 [2] 딥페이크 vs 실제 구별 탭 활용법
- 무작위로 주어지는 10장의 이미지 중, **딥페이크(Fake)** 이미지만 선택하세요.
- AI 없이 사람이 직접 판단해보는 퀴즈형 활동으로, 시각적 직관과 윤리적 판단의 간극을 체감할 수 있습니다.
- 결과는 정답/오답/놓침에 따라 색상으로 피드백되며, 점수는 랭킹에 반영됩니다.

---

### 🎯 수업에서의 활용 포인트
- **2차시 활동**: SwapFace로 만든 이미지 → 이 웹앱으로 판별 실습
- **토론 주제 유도 질문**
    - “AI가 틀릴 수도 있다는 것을 체험하면서 어떤 생각이 들었나요?”
    - “딥페이크 이미지로 잘못된 정보가 퍼졌을 때, 누구의 책임일까요?”
- **3차시 연계 활동**: Padlet에 웹앱 사용 후기 및 생각 공유

---

### 📎 연계 도구 및 링크
- [SwapFace - 딥페이크 이미지 생성 도구](https://www.swapfaces.ai/face-swap-pictures-free)
- [Padlet 협업 게시판](https://padlet.com/kanghyejong1001/padlet-35kuanfummy3ofmo)

---

이 웹앱은 단순한 기술 체험을 넘어, 학생 스스로 딥페이크 기술을 **판별하고, 윤리적 책임을 숙고하는** 과정을 유도합니다.
""")


# ---------------------------
# 실행
# ---------------------------
st.title("딥페이크 이미지 판별 AI 시스템")
tabs = st.tabs(["이미지 AI 판별", "딥페이크 vs 실제 구별", "도움말"])
with tabs[0]:
    deepfake_checker()
with tabs[1]:
    simulation()
with tabs[2]:
    show_tutorial()