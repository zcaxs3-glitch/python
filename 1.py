import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st


st.title("파이썬 그래프 대시보드")
st.write("버튼을 누르면 파이썬 코드가 실행되어 그래프를 표시합니다.")
if st.button("🚀 코드 작동 시작"):
    with st.spinner("그래프 생성 중..."):

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
        
                e     = d - y
                delta = y*(1-y) * e
        
                dW = alpha*delta*x
        
                W[0][0] = W[0][0] + dW[0]
                W[0][1] = W[0][1] + dW[1]
                W[0][2] = W[0][2] + dW[2]
        
            return W

        def TestDeltaSGD():
            X = np.array([[0, 0, 1],
                          [0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    
            D = np.array([[0],
                          [0],
                          [1],
                          [1]])
        
            W = 2*np.random.random((1, 3)) - 1
    
            W_list = [[],[],[]]
        
            for _epoch in range(10000):
                W = DeltaSGD(W, X, D)        
                W_list[0].append(W[0][0])
                W_list[1].append(W[0][1])
                W_list[2].append(W[0][2])
        
    # print(W_list)
                fig, ax = plt.subplots(1, 1)
                fig.subplots_adjust(hspace = 0.4)
    
    # t = np.arange(0., 10000., 1)
    # ax.plot(W_list[0], 'b')
            ax.plot(W_list[0], 'b',
                    W_list[1], 'r',
                    W_list[2], 'g')
            ax.set_title('weight epoch update', fontsize = 20)     
            plt.show()     
     
            N = 4
            for k in range(N):
                x = X[k,:].T
                v = np.matmul(W, x)
                y = Sigmoid(v)
                print(y)
    


        if __name__ == '__main__':
            TestDeltaSGD()
            st.success("실행 완료!")
            st.pyplot(plt)
            plt.close('all')
            
