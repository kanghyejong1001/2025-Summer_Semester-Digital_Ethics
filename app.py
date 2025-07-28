import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import random

# ---------------------------
# 사이드바 소개
# ---------------------------
st.sidebar.title("웹앱 소개")
st.sidebar.markdown("""
이 웹앱은 AI 모델을 활용하여 이미지가 딥페이크(Fake)인지 실제(Real)인지 판별하고, 
결과를 통해 딥페이크 기술의 위험성과 활용의 적절성을 판단할 수 있도록 돕는 도구입니다.

이 활동을 통해 고등학생들은 다음과 같은 역량을 기를 수 있습니다:
- 컴퓨팅 사고력: 문제 분해, 패턴 인식
- 인공지능 소양: AI의 원리와 한계 이해
- 디지털 문화 소양: 딥페이크의 사회적 영향에 대한 인식

---
### 탭별 소개

**[이미지 판별]**
- 업로드한 얼굴 이미지가 딥페이크인지 AI가 판별합니다.
- 예측 결과(딥페이크/실제), 확률, 이유가 제공됩니다.
- 이미지 사용 허락 및 정답 체크가 포함됩니다.

**[딥페이크 vs 실제 구별]**
- 다양한 이미지를 무작위로 제시하고 직접 구별해보는 시뮬레이션입니다.
- 선택 후 정답을 확인하고 점수를 누적할 수 있습니다.

**[도움말]**
- 웹앱 전반의 사용법과 교육적 활용 방안을 안내합니다.
""")

# ---------------------------
# AI 예측 (간단한 샘플 모델 시뮬레이션)
# ---------------------------
def predict_image(image):
    import torchvision.transforms as transforms
    import torch
    from torchvision import models

    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0)  # 배치 차원 추가

    # 사전 학습된 ResNet18 모델 사용
    model = models.resnet18(pretrained=True)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)

    confidence, predicted_idx = torch.max(prob, 0)
    prediction = "Fake" if predicted_idx.item() % 2 == 0 else "Real"  # 예시 기준
    probability = round(confidence.item(), 2)
    reason = f"ResNet18 기반 분류 결과, 클래스 #{predicted_idx.item()}에서 높은 확률 감지됨."
    return prediction, probability, reason


# ---------------------------
# 딥페이크 판별 탭
# ---------------------------
def deepfake_checker():
    st.subheader("이미지 판별 (AI 분석)")
    st.markdown("""
    **사용 방법:**
    - 아래에 얼굴 이미지를 업로드하세요.
    - '제출' 버튼을 누르면 AI가 분석 결과를 알려줍니다.
    - 결과에는 예측, 확률, 판단 이유가 포함됩니다.
    - 이후 이미지 사용 동의 여부를 선택할 수 있습니다.
    """)

    uploaded_file = st.file_uploader("이미지를 업로드 해주세요.", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", width=256)

        if st.button("제출"):
            prediction, prob, reason = predict_image(image)
            st.success(f"예측 결과: {prediction}")
            st.info(f"판별 확률: {prob*100}%")
            st.write(f"판단 이유: {reason}")

            st.markdown("---")
            st.subheader("사용자 응답")
            actual = st.radio("위 이미지는 딥페이크 vs 실제 무엇인가요? (정답을 알고 있다면 선택해주세요.)", ["Real", "Fake"])
            consent = st.radio("이 이미지를 웹앱에서 사용하는 것에 동의하시나요?", ["동의", "미동의"])

            if st.button("응답 제출"):
                if consent == "동의":
                    save_dir = f"user_images/{actual.lower()}"
                    image_path = os.path.join(save_dir, uploaded_file.name)
                    image.save(image_path)
                    st.success("이미지와 정보가 저장되었습니다. 시뮬레이션 탭에서 사용됩니다.")
                else:
                    st.warning("사용 동의를 하지 않아 저장되지 않았습니다.")


# ---------------------------
# 딥페이크 vs 실제 시뮬레이션
# ---------------------------
def simulation():
    import zipfile

    st.subheader("🎮 딥페이크 vs 실제 구별")
    st.markdown("""
    **게임 설명:**
    - 실제 딥페이크 이미지와 실제 이미지를 활용해 AI 퀴즈를 진행합니다.
    - 정답을 맞히면 점수가 올라가고, 다음 문제로 계속 진행됩니다!
    """)

    # 전체 통계 저장용 세션 상태 초기화
    if 'total_correct' not in st.session_state:
        st.session_state.total_correct = 0
        st.session_state.total_attempt = 0
        st.session_state.scoreboard = []

    username = st.text_input("**닉네임을 입력하세요** (랭킹 등록용)", value="익명 사용자")

    # 외부 데이터셋 업로드 (zip 파일)
    st.markdown("#### 📥 Kaggle 등에서 받은 데이터셋(zip) 업로드")
    dataset_zip = st.file_uploader("외부 데이터셋 압축 파일을 업로드하세요 (Real/Fake 폴더 구조)", type="zip")
    external_files = []
    if dataset_zip:
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall("external_dataset")
        for label in ["real", "fake"]:
            folder = f"external_dataset/{label}"
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    external_files.append({"filename": f, "label": label.capitalize(), "folder": folder})

    # 사용자 동의 이미지
    user_files = []
    for label in ["real", "fake"]:
        folder = f"user_images/{label}"
        for f in os.listdir(folder):
            user_files.append({"filename": f, "label": label.capitalize(), "folder": folder})

    # 관리자 필터링 페이지
    st.markdown("---")
    with st.expander("🔧 관리자 이미지 관리 및 동의 철회"):
        st.markdown("**동의된 사용자 이미지 보기 및 철회**")
        for label in ["real", "fake"]:
            folder = f"user_images/{label}"
            files = os.listdir(folder)
            for file in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"`{label.upper()}` - {file}")
                with col2:
                    if st.button("❌ 삭제", key=f"{label}_{file}"):
                        os.remove(os.path.join(folder, file))
                        st.warning(f"{file} 삭제됨")

    all_data = user_files + external_files
    if not all_data:
        st.warning("사용자가 동의한 이미지나 외부 데이터셋이 없어 시뮬레이션을 진행할 수 없습니다.")
        return

    combined = pd.DataFrame(all_data)
    combined = combined.sample(frac=1).reset_index(drop=True)
    total_images = len(combined)

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        st.session_state.score = 0
        st.session_state.total = 0
        st.session_state.answer_given = False

    idx = st.session_state.current_index
    if idx >= total_images:
        st.success(f"🎉 모든 문제를 완료했습니다! 최종 점수: {st.session_state.score}/{st.session_state.total}")
        st.session_state.total_correct += st.session_state.score
        st.session_state.total_attempt += st.session_state.total
        st.session_state.scoreboard.append({
            "user": username,
            "score": st.session_state.score,
            "total": st.session_state.total
        })

        st.markdown("---")
        st.markdown("### 📊 전체 통계")
        accuracy = st.session_state.total_correct / st.session_state.total_attempt * 100 if st.session_state.total_attempt > 0 else 0.0
        st.metric("누적 정답률", f"{accuracy:.1f}%")
        st.metric("전체 정답 수", st.session_state.total_correct)
        st.metric("전체 시도 수", st.session_state.total_attempt)

        st.markdown("### 🏆 랭킹")
        if st.session_state.scoreboard:
            df_rank = pd.DataFrame(st.session_state.scoreboard)
            df_rank['accuracy'] = df_rank['score'] / df_rank['total'] * 100
            df_rank = df_rank.sort_values(by=['score', 'accuracy'], ascending=False).reset_index(drop=True)
            st.dataframe(df_rank[['user', 'score', 'total', 'accuracy']].rename(columns={
                'user': '닉네임', 'score': '점수', 'total': '시도 수', 'accuracy': '정답률 (%)'
            }))
        else:
            st.info("아직 등록된 랭킹 정보가 없습니다.")

        if st.button("🔄 다시 시작"):
            st.session_state.current_index = 0
            st.session_state.score = 0
            st.session_state.total = 0
            st.session_state.answer_given = False
        return

    row = combined.loc[idx]
    img_path = os.path.join(row.folder, row.filename)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_path, width=220, caption=f"사진 {idx + 1} / {total_images}")

    with col2:
        choice = st.radio("이 이미지는 어떤가요?", ["Real", "Fake"], key=idx)
        if st.button("✅ 정답 확인"):
            if choice == row.label:
                st.success("🎯 정답입니다!")
                st.session_state.score += 1
            else:
                st.error(f"❌ 오답입니다. 정답은 {row.label}입니다.")
            st.session_state.total += 1
            st.session_state.answer_given = True

        if st.session_state.answer_given:
            if st.button("➡️ 다음 문제"):
                st.session_state.current_index += 1
                st.session_state.answer_given = False

    st.markdown("---")
    st.markdown("### 📊 전체 통계")
    if st.session_state.total_attempt > 0:
        accuracy = st.session_state.total_correct / st.session_state.total_attempt * 100
        st.metric("누적 정답률", f"{accuracy:.1f}%")
        st.metric("전체 정답 수", st.session_state.total_correct)
        st.metric("전체 시도 수", st.session_state.total_attempt)
    else:
        st.info("아직 집계된 통계가 없습니다.")

    st.markdown("### 🏆 랭킹")
    if st.session_state.scoreboard:
        df_rank = pd.DataFrame(st.session_state.scoreboard)
        df_rank['accuracy'] = df_rank['score'] / df_rank['total'] * 100
        df_rank = df_rank.sort_values(by=['score', 'accuracy'], ascending=False).reset_index(drop=True)
        st.dataframe(df_rank[['user', 'score', 'total', 'accuracy']].rename(columns={
            'user': '닉네임', 'score': '점수', 'total': '시도 수', 'accuracy': '정답률 (%)'
        }))
    else:
        st.info("아직 등록된 랭킹 정보가 없습니다.")


# ---------------------------
# 도움말
# ---------------------------
def show_tutorial():
    st.subheader("도움말")
    st.markdown("""
    이 웹앱은 학생들이 인공지능을 통해 딥페이크와 실제 이미지를 구별할 수 있도록 돕습니다. 
    학습자는 이를 통해 컴퓨팅 사고력과 AI 소양을 기르게 됩니다.

    **활용 예시:**
    - 윤리 수업: 딥페이크의 사회적 영향 논의
    - 정보 수업: AI 이미지 분석 체험
    - 융합 수업: 미디어, 사회, 과학과의 연계 가능

    **문제 해결력 향상을 위한 팁:**
    - 얼굴의 윤곽, 눈동자, 피부결, 배경의 이상 유무를 관찰해 보세요.
    - 반복적인 시뮬레이션을 통해 패턴을 익힐 수 있습니다.
    """)

# ---------------------------
# Streamlit 앱 실행
# ---------------------------
st.title("딥페이크 이미지 판별 AI 시스템")
tabs = st.tabs(["이미지 판별", "딥페이크 vs 실제 구별", "도움말"])

with tabs[0]:
    deepfake_checker()
with tabs[1]:
    simulation()
with tabs[2]:
    show_tutorial()
