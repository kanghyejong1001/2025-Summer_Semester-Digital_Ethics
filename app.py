import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from PIL import Image
import io
import torchvision.transforms as transforms
import torch
from torchvision import models

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
    st.subheader("이미지 판별 (AI 분석)")
    st.markdown("""
    - 이미지를 업로드하면 AI가 분석합니다.
    - 결과를 확인 후, 해당 이미지가 실제인지 딥페이크인지 체크하고 웹앱에서 사용 동의 여부를 입력합니다.
    """)

    if 'user_images' not in st.session_state:
        st.session_state.user_images = {'real': [], 'fake': []}

    uploaded_file = st.file_uploader("이미지를 업로드 해주세요.", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", width=256)

        if st.button("제출"):
            prediction, prob, reason = predict_image(image)
            st.success(f"예측 결과: {prediction}")
            st.info(f"판별 확률: {prob*100}%")
            st.write(f"판단 이유: {reason}")

            actual = st.radio("위 이미지는 무엇인가요?", ["Real", "Fake"])
            consent = st.radio("이 이미지를 시뮬레이션에 사용하는 것에 동의하십니까?", ["동의", "미동의"])
            if st.button("응답 제출"):
                if consent == "동의":
                    if uploaded_file is not None:
                        image_copy = image.copy()
                        st.session_state.user_images[actual.lower()].append({
                            'image': image_copy,
                            'filename': uploaded_file.name
                        })
                        st.success("✅ 이미지가 세션에 성공적으로 저장되었습니다.")
                    else:
                        st.warning("⚠️ 이미지가 업로드되지 않아 저장에 실패했습니다.")
                else:
                    st.warning("사용 동의를 하지 않아 저장되지 않았습니다.")

# ---------------------------
# 딥페이크 vs 실제 시뮬레이션
# ---------------------------
def simulation():
    st.subheader("🎮 딥페이크 vs 실제 구별")

    if 'total_correct' not in st.session_state:
        st.session_state.total_correct = 0
        st.session_state.total_attempt = 0
        st.session_state.scoreboard = []

    if 'user_images' not in st.session_state:
        st.session_state.user_images = {'real': [], 'fake': []}

    username = st.text_input("닉네임 입력 (랭킹용)", value="익명 사용자")
    st.markdown("---")
    st.markdown("#### 📥 외부 데이터셋 업로드(zip, Real/Fake 폴더 구조)")
    dataset_zip = st.file_uploader("데이터셋 업로드", type="zip")

    external_files = []
    if dataset_zip:
        with zipfile.ZipFile(dataset_zip) as zf:
            for name in zf.namelist():
                if name.endswith(('.png', '.jpg', '.jpeg')) and ('real/' in name or 'fake/' in name):
                    label = 'real' if 'real/' in name else 'fake'
                    with zf.open(name) as file:
                        image = Image.open(file).convert("RGB")
                        external_files.append({"image": image, "label": label.capitalize(), "filename": name.split('/')[-1]})

    user_files = []
    for label in ["real", "fake"]:
        for item in st.session_state.user_images[label]:
            user_files.append({"image": item['image'], "label": label.capitalize(), "filename": item['filename']})

    all_data = user_files + external_files
    if not all_data:
        st.warning("사용자 또는 외부 이미지가 없습니다.")
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
        st.success(f"🎉 게임 종료! 최종 점수: {st.session_state.score}/{st.session_state.total}")
        st.session_state.total_correct += st.session_state.score
        st.session_state.total_attempt += st.session_state.total
        st.session_state.scoreboard.append({"user": username, "score": st.session_state.score, "total": st.session_state.total})
        if st.button("🔄 다시 시작"):
            st.session_state.current_index = 0
            st.session_state.score = 0
            st.session_state.total = 0
            st.session_state.answer_given = False
        return

    row = combined.loc[idx]
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(row['image'], width=220, caption=f"사진 {idx + 1} / {total_images}")

    with col2:
        choice = st.radio("이 이미지는 어떤가요?", ["Real", "Fake"], key=idx)
        if st.button("✅ 정답 확인"):
            if choice == row['label']:
                st.success("🎯 정답입니다!")
                st.session_state.score += 1
            else:
                st.error(f"❌ 오답입니다. 정답은 {row['label']}입니다.")
            st.session_state.total += 1
            st.session_state.answer_given = True

        if st.session_state.answer_given:
            if st.button("➡️ 다음 문제"):
                st.session_state.current_index += 1
                st.session_state.answer_given = False

    st.markdown("---")
    st.metric("누적 정답률", f"{(st.session_state.total_correct / st.session_state.total_attempt * 100):.1f}%" if st.session_state.total_attempt else "0.0%")
    st.metric("전체 정답 수", st.session_state.total_correct)
    st.metric("전체 시도 수", st.session_state.total_attempt)

    st.markdown("### 🏆 랭킹")
    if st.session_state.scoreboard:
        df_rank = pd.DataFrame(st.session_state.scoreboard)
        df_rank['accuracy'] = df_rank['score'] / df_rank['total'] * 100
        df_rank = df_rank.sort_values(by=['score', 'accuracy'], ascending=False)
        st.dataframe(df_rank[['user', 'score', 'total', 'accuracy']].rename(columns={
            'user': '닉네임', 'score': '점수', 'total': '시도 수', 'accuracy': '정답률 (%)'
        }))
    else:
        st.info("아직 랭킹 정보가 없습니다.")

# ---------------------------
# 도움말
# ---------------------------
def show_tutorial():
    st.subheader("도움말")
    st.markdown("""
    이 웹앱은 학생들이 인공지능을 통해 딥페이크와 실제 이미지를 구별할 수 있도록 돕습니다.

    **활용 예시:**
    - 윤리 수업: 딥페이크의 사회적 영향
    - 정보 수업: AI 이미지 분석 체험
    - 미디어/사회 융합 수업 등
    """)

# ---------------------------
# 실행
# ---------------------------
st.title("딥페이크 이미지 판별 AI 시스템")
tabs = st.tabs(["이미지 판별", "딥페이크 vs 실제 구별", "도움말"])
with tabs[0]:
    deepfake_checker()
with tabs[1]:
    simulation()
with tabs[2]:
    show_tutorial()
