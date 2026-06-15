import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# --- 1. 순수 수학 함수들 (최상위 배치) ---
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def DeltaSGD(W, X, D):
    alpha = 0.9
    N = 4
    for k in range(N):
        x = X[k, :].T
        d = D[k]

        v = np.matmul(W, x)
        y = Sigmoid(v)

        e = d - y
        delta = y * (1 - y) * e

        dW = alpha * delta * x

        W[0][0] = W[0][0] + dW[0]
        W[0][1] = W[0][1] + dW[1]
        W[0][2] = W[0][2] + dW[2]

    return W


# --- 2. ⚡ 마법의 캐싱 함수: 10,000번 수치 계산만 수행 ---
@st.cache_data
def TestDeltaSGD_cached():
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    D = np.array([[0], [0], [1], [1]])

    # 고정된 결과를 위해 무작위 가중치 대신 시드 고정 혹은 고정 배치
    # 캐싱 함수 내부의 무작위 값은 최초 1회만 고정 저장됩니다.
    W = 2 * np.random.random((1, 3)) - 1
    W_list = [[], [], []]

    # 10,000번 돌며 '숫자 데이터'만 리스트에 수집 (여기선 그래프 안 그림!)
    for _epoch in range(10000):
        W = DeltaSGD(W, X, D)
        W_list[0].append(W[0][0])
        W_list[1].append(W[0][1])
        W_list[2].append(W[0][2])

    # 계산 결과 데이터만 밖으로 돌려줌
    return W_list


# --- 3. 웹 화면 UI 구성 ---
st.title("파이썬 그래프 대시보드")
st.write("버튼을 누르면 파이썬 코드가 실행되어 그래프를 표시합니다.")

if st.button("🚀 코드 작동 시작"):
    with st.spinner("그래프 생성 중..."):

        # [캐시 작동] 최초 1회만 10,000번 연산하고, 이후부턴 0초 만에 데이터만 쏙 빼옴
        W_list = TestDeltaSGD_cached()

        # 🎨 그래프 그리기는 캐시 밖에서 '딱 1번만' 실행! (1만 번 돌던 거 꺼냄)
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0.4)

        ax.plot(W_list[0], "b", label="W[0]")
        ax.plot(W_list[1], "r", label="W[1]")
        ax.plot(W_list[2], "g", label="W[2]")

        ax.set_title("weight epoch update", fontsize=20)
        ax.legend()  # 어떤 선이 어떤 가중치인지 라벨 표시

        # Streamlit 화면에 안전하게 그래프 출력
        st.success("실행 완료!")
        st.pyplot(fig)

        # 🧹 메모리 청소 (무료 서버 폭발 방지)
        plt.close(fig)
