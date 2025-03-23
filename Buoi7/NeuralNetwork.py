import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os
import mlflow
import mlflow.keras
import random
from datetime import datetime
import matplotlib.pyplot as plt
import traceback
import time
import requests
from mlflow.exceptions import MlflowException
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Tắt GPU để tránh lỗi cuDNN/cuBLAS (tạm thời) và tắt oneDNN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Chỉ dùng CPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Tắt thông báo oneDNN

# Hàm khởi tạo MLflow
def mlflow_input():
    try:
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/npbthang/Mnist.mlflow"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
        os.environ["MLFLOW_TRACKING_USERNAME"] = "npbthang"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "6ad5ad3cc6d4b2f9efb9f28b1aa13618d2ce7357"  # Cập nhật token nếu cần
        mlflow.set_experiment("Neural_Network")
        st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
        st.success("✅ MLflow được khởi tạo thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi khởi tạo MLflow: {str(e)}")
        traceback.print_exc()

# Hàm kiểm tra kết nối MLflow
def check_mlflow_connection():
    try:
        response = requests.get(st.session_state['mlflow_url'], timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Hàm tải dữ liệu từ OpenML hoặc file cục bộ
@st.cache_data
def load_mnist_data():
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.uint8)
        return X, y
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu MNIST từ OpenML: {str(e)}")
        return None, None
# Tab hiển thị dữ liệu
def data():
    st.header("📘 Dữ Liệu MNIST từ OpenML")
    
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.X = None
        st.session_state.y = None

    if st.button("⬇️ Tải dữ liệu từ OpenML"):
        with st.spinner("⏳ Đang tải dữ liệu MNIST từ OpenML..."):
            X, y = load_mnist_data()
            if X is not None and y is not None:
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success("✅ Dữ liệu đã được tải thành công!")
            else:
                st.error("❌ Không thể tải dữ liệu!")

    if st.session_state.data_loaded:
        X, y = st.session_state.X, st.session_state.y
        st.write(f"""
            **Thông tin tập dữ liệu MNIST:**
            - Tổng số mẫu: {X.shape[0]}
            - Kích thước mỗi ảnh: 28 × 28 pixels (784 đặc trưng)
            - Số lớp: 10 (chữ số từ 0-9)
        """)

        st.subheader("Một số hình ảnh mẫu")
        n_samples = 10
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Label: {y[idx]}")
            axes[row, col].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# Tab lý thuyết Neural Network
def explain_nn():
    st.markdown("""
    ## 🧠 Neural Network Cơ Bản

    **Neural Network (Mạng nơ-ron nhân tạo - ANN)** là một mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của não bộ con người. Mạng bao gồm nhiều nơ-ron nhân tạo kết nối với nhau thành các lớp (layers), giúp mô hình học và nhận diện các mẫu trong dữ liệu.

    ### 🔰 Kiến trúc cơ bản:
    ### 📌 Cấu trúc của một mạng nơ-ron nhân tạo gồm ba loại lớp chính:
    1. **Input Layer**: Lớp tiếp nhận dữ liệu đầu vào.
    2. **Hidden Layers**: Xử lý thông tin thông qua các trọng số (weights) và hàm kích hoạt.
    3. **Output Layer**: Lớp đưa ra kết quả dự đoán.
    """)
    
    # Ảnh 1
    st.image("https://i0.wp.com/nttuan8.com/wp-content/uploads/2019/03/nn-1.png?resize=768%2C631&ssl=1", 
         caption="Cấu trúc mạng nơ-ron (Nguồn: [nttuan8.com](https://nttuan8.com/bai-3-neural-network/))")
    
    st.markdown("""
    ## 📌 Công thức toán học trong Neural Network:
    Mỗi nơ-ron trong một lớp nhận tín hiệu từ các nơ-ron lớp trước, nhân với trọng số (**weights**), cộng với **bias**, rồi đưa vào một **hàm kích hoạt** để quyết định tín hiệu truyền đi.
    """)

    st.markdown("### 🎯 Công thức tính giá trị đầu ra của một nơ-ron:")
    st.latex(r" z = \sum_{i=1}^{n} w_i x_i + b ")

    st.markdown(r"""
    Trong đó:
    - $$ x_i $$ là đầu vào (**input features**).
    - $$ w_i $$ là **trọng số** (**weights**) kết nối với nơ-ron đó.
    - $$ b $$ là **bias** (hệ số dịch chuyển).
    - $$ z $$ là tổng có trọng số (**weighted sum**).

    Sau khi tính toán $$ z $$, nó sẽ đi qua một **hàm kích hoạt** $$ f(z) $$ để tạo ra giá trị đầu ra.
    """)

    st.markdown("""
    ### 🎯 Hàm Kích Hoạt (Activation Functions)
    Hàm kích hoạt giúp mạng học được các tính phi tuyến tính, nhờ đó có thể mô hình hóa các mối quan hệ phức tạp.
    """)
    
    st.markdown("- **Sigmoid:** Chuyển đổi giá trị đầu vào thành khoảng từ 0 đến 1, phù hợp cho bài toán phân loại nhị phân.")
    st.latex(r"f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}")
    # Ảnh 2
    st.image("https://images.viblo.asia/1489e092-5b68-4c75-834a-1a2148460759.png", 
         caption="Hàm kích hoạt trong mạng nơ-ron (Nguồn: [viblo.asia](https://viblo.asia/p/tai-sao-lai-su-dung-activation-function-trong-neural-network-MG24BwweJz3))")

    st.markdown("- **Tanh (Hyperbolic Tangent):** Đầu ra nằm trong khoảng từ -1 đến 1, giúp xử lý dữ liệu có cả giá trị dương và âm.")
    st.latex(r"f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")
    st.image("https://images.viblo.asia/54ac7d4c-2639-4ec3-9644-ce489210819a.png", 
         caption="Hàm kích hoạt trong mạng nơ-ron (Nguồn: [viblo.asia](https://viblo.asia/p/tai-sao-lai-su-dung-activation-function-trong-neural-network-MG24BwweJz3))")


    st.markdown("- **ReLU (Rectified Linear Unit):** Nếu đầu vào âm thì bằng 0, còn nếu dương thì giữ nguyên giá trị.")
    st.latex(r"f(z) = ReLU(z) = \max(0, z)")
    st.image("https://images.viblo.asia/38602515-6466-486e-8bfa-990951ce61b6.png", 
         caption="Hàm kích hoạt trong mạng nơ-ron (Nguồn: [viblo.asia](https://viblo.asia/p/tai-sao-lai-su-dung-activation-function-trong-neural-network-MG24BwweJz3))", )

    st.markdown("### 🔄 Tại sao sử activation function lại cần thiết.")
    st.markdown("1️ Giữ các giá trị output trong khoảng nhất định:")
    st.markdown("với một model với hàng triệu tham số thì kết quả của phép nhân tuyến tính từ phương trình (1) sẽ có thể là một giá trị rất lớn (dương vô cùng) hoặc rất bé (âm vô cùng) và có thể gây ra những vấn đề về mặt tính toán và mạng rất khó để có thể hội tụ."
    " Việc sử dụng activation có thể giới hạn đầu ra ở một khoảng giá trị nào đó, ví dụ như hàm sigmoid,softmax giới hạn giá trị đầu ra trong khoảng (0, 1) cho dù kết quả của phép nhân tuyến tính là bao nhiêu đi chăng nữa.")

    st.markdown("#### 🔄 Tính toán loss")
    st.markdown("- Hàm mất mát đo lường sai số giữa dự đoán và thực tế.")
    st.latex(r"L = - \sum y_{true} \log(y_{pred})")  # Cross-Entropy Loss









   # st.markdown("#### 🔄 Thuật Toán Tối Ưu")
    #st.markdown("- Thuật toán tối ưu là cơ sở để xây dựng mô hình neural network với mục đích học được các features ( hay pattern) của dữ liệu đầu vào, "
    #"từ đó có thể tìm 1 cặp weights và bias phù hợp để tối ưu hóa model.")
    #st.markdown("- **Adam:** Một trong những thuật toán tối ưu phổ biến cho Neural Network.")







# Tab chia dữ liệu
def split_data():
    st.header("📌 Chia dữ liệu Train/Validation/Test")
    if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
        st.warning("⚠ Vui lòng tải dữ liệu từ tab 'Dữ Liệu' trước khi tiếp tục!")
        return

    X, y = st.session_state.X, st.session_state.y
    total_samples = X.shape[0]
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False

    num_samples = st.slider("📌 Chọn số lượng ảnh để train:", 1000, total_samples, 10000)
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        try:
            indices = np.random.choice(total_samples, num_samples, replace=False)
            X_selected = X[indices]
            y_selected = y[indices]

            stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
            )

            stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size/(100 - test_size),
                stratify=stratify_option, random_state=42
            )

            st.session_state.total_samples = num_samples
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.val_size = X_val.shape[0]
            st.session_state.train_size = X_train.shape[0]
            st.session_state.data_split_done = True

            summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
            })
            st.success("✅ Dữ liệu đã được chia thành công!")
            st.table(summary_df)
        except Exception as e:
            st.error(f"❌ Lỗi khi chia dữ liệu: {str(e)}")
            traceback.print_exc()

    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")


# Tab huấn luyện

def train():
    st.header("⚙️ Huấn luyện Neural Network")
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # Lấy dữ liệu đã chia từ session_state
    X_train = st.session_state.X_train
    X_val = st.session_state.X_val
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val = st.session_state.y_val
    y_test = st.session_state.y_test

    # Chuẩn hóa dữ liệu (nếu chưa chuẩn hóa)
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_val = X_val.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    # Các tham số huấn luyện
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5, key="train_k_folds")
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2, key="train_num_layers")
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32, key="train_num_neurons")
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"], key="train_activation")
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"], key="train_optimizer")
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1, key="train_epochs")
    learning_rate = st.number_input(
        "⚡ Tốc độ học (Learning Rate):", 
        min_value=1e-5, 
        max_value=1e-1, 
        value=1e-3, 
        step=1e-5, 
        format="%.5f", 
        key="train_learning_rate"
    )
    loss_fn = "sparse_categorical_crossentropy"  # Hàm mất mát cố định

    run_name = st.text_input("🔹 Nhập tên Run:", "", key="train_run_name")
    st.session_state["run_name"] = run_name if run_name else "Default_NN_Run"

    if "training_results" not in st.session_state:
        st.session_state.training_results = None

    # Khởi tạo biến lưu accuracy và loss nếu chưa có
    if "fold_accuracies" not in st.session_state:
        st.session_state.fold_accuracies = []
    if "fold_losses" not in st.session_state:
        st.session_state.fold_losses = []
    if "test_accuracy" not in st.session_state:
        st.session_state.test_accuracy = None

    if st.button("Huấn luyện mô hình", key="train_button"):
        if not run_name:
            st.error("⚠️ Vui lòng nhập tên Run trước khi huấn luyện!")
            return
        
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}") as run:
            st.write(f"Debug: Run Name trong MLflow: {run.info.run_name}")

            # Log các tham số
            mlflow.log_param("total_samples", st.session_state.total_samples)
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("validation_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("k_folds", k_folds)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("num_neurons", num_neurons)
            mlflow.log_param("activation", activation)
            mlflow.log_param("optimizer", optimizer)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)

            st.write("⏳ Đang đánh giá và huấn luyện mô hình...")
            progress_bar = st.progress(0)

            try:
                # Cross-validation với KFold
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                accuracies = []
                losses = []  # Khởi tạo danh sách losses
                fold_count = 0

                for train_idx, val_idx in kf.split(X_train, y_train):
                    X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
                    y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

                    # Xây dựng mô hình Keras
                    model = Sequential()
                    model.add(Input(shape=(X_k_train.shape[1],)))
                    for _ in range(num_layers):
                        model.add(Dense(num_neurons, activation=activation))
                    model.add(Dense(10, activation="softmax"))

                    # Chọn optimizer
                    if optimizer == "adam":
                        opt = Adam(learning_rate=learning_rate)
                    elif optimizer == "sgd":
                        opt = SGD(learning_rate=learning_rate)
                    else:
                        opt = RMSprop(learning_rate=learning_rate)

                    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                    # Huấn luyện mô hình
                    history = model.fit(X_k_train, y_k_train, epochs=epochs, 
                                      validation_data=(X_k_val, y_k_val), verbose=0)

                    # Đánh giá trên fold hiện tại
                    val_loss, val_accuracy = model.evaluate(X_k_val, y_k_val, verbose=0)
                    accuracies.append(val_accuracy)
                    losses.append(val_loss)  # Thêm val_loss vào danh sách losses

                    fold_count += 1
                    progress_bar.progress(fold_count / k_folds)
                    st.write(f"📌 Fold {fold_count}/{k_folds} - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")

                # Lưu accuracies và losses của các fold
                st.session_state.fold_accuracies = accuracies
                st.session_state.fold_losses = losses
                mean_cv_accuracy = np.mean(accuracies)
                mean_cv_loss = np.mean(losses)  # Tính mean_cv_loss từ losses

                # Đánh giá trên tập test
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.session_state.test_accuracy = test_accuracy
                progress_bar.progress(1.0)

                # Log metrics vào MLflow
                mlflow.log_metric("cv_accuracy_mean", mean_cv_accuracy)
                mlflow.log_metric("cv_loss_mean", mean_cv_loss)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_loss", test_loss)

                # Lưu mô hình vào MLflow
                mlflow.keras.log_model(model, "neural_network")

                # Lưu kết quả vào session_state
                st.session_state.training_results = {
                    "cv_accuracy_mean": mean_cv_accuracy,
                    "cv_loss_mean": mean_cv_loss,
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "success"
                }

                # Lưu mô hình vào danh sách models
                if "models" not in st.session_state:
                    st.session_state["models"] = []
                model_name = "neural_network"
                full_run_name = f"Train_{st.session_state['run_name']}"
                existing_model = next((item for item in st.session_state["models"] if item["name"] == model_name), None)
                if existing_model:
                    count = 1
                    new_model_name = f"{model_name}_{count}"
                    while any(item["name"] == new_model_name for item in st.session_state["models"]):
                        count += 1
                        new_model_name = f"{model_name}_{count}"
                    model_name = new_model_name
                    st.warning(f"⚠️ Mô hình được lưu với tên: {model_name}")
                st.session_state["models"].append({
                    "name": model_name,
                    "run_name": full_run_name,
                    "model": model
                })

            except Exception as e:
                error_message = str(e)
                mlflow.log_param("status", "failed")
                mlflow.log_metric("cv_accuracy_mean", -1)
                mlflow.log_metric("cv_loss_mean", -1)
                mlflow.log_metric("test_accuracy", -1)
                mlflow.log_metric("test_loss", -1)
                mlflow.log_param("error_message", error_message)
                st.session_state.training_results = {
                    "error_message": error_message,
                    "run_name": f"Train_{st.session_state['run_name']}",
                    "status": "failed"
                }

    # Hiển thị kết quả sau khi huấn luyện (ở ngoài khối if)
    if "fold_accuracies" in st.session_state and st.session_state.fold_accuracies:
        st.subheader("📊 Kết quả huấn luyện")
        for i, (acc, loss) in enumerate(zip(st.session_state.fold_accuracies, st.session_state.fold_losses), 1):
            st.write(f"📌 Fold {i}/{len(st.session_state.fold_accuracies)} - Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        if st.session_state.test_accuracy is not None:
            st.write(f"✅ Test Accuracy: {st.session_state.test_accuracy:.4f}")
        if st.session_state.training_results and st.session_state.training_results["status"] == "success":
            st.success(f"📊 Cross-Validation Accuracy trung bình: {st.session_state.training_results['cv_accuracy_mean']:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state.training_results['run_name']}**!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
        elif st.session_state.training_results and st.session_state.training_results["status"] == "failed":
            st.error(f"❌ Lỗi khi huấn luyện mô hình: {st.session_state.training_results['error_message']}")

    # Hiển thị danh sách mô hình đã lưu
    if "models" in st.session_state and st.session_state["models"]:
        st.write(f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")
        st.write("📋 Danh sách các mô hình đã lưu:")
        model_display = [f"{model['run_name']} ({model['name']})" for model in st.session_state["models"]]
        st.write(", ".join(model_display))
def du_doan():
    st.header("✍️ Dự đoán số viết tay")
    
    # Kiểm tra xem có mô hình nào trong st.session_state["models"] không
    if "models" not in st.session_state or not st.session_state["models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước!")
        return

    # Lấy danh sách các mô hình đã huấn luyện
    model_options = [f"{m['run_name']} ({m['name']})" for m in st.session_state["models"]]
    selected_model_name = st.selectbox("📋 Chọn mô hình để dự đoán:", model_options, key="predict_model_select")
    
    # Lấy mô hình được chọn từ danh sách
    selected_model = next(m["model"] for m in st.session_state["models"] if f"{m['run_name']} ({m['name']})" == selected_model_name)
    st.success(f"✅ Đã chọn mô hình: {selected_model_name}")

    input_method = st.radio("📥 Chọn phương thức nhập liệu:", ("Vẽ tay", "Tải ảnh lên"), key="predict_input_method")

    img = None
    if input_method == "Vẽ tay":
        if "key_value" not in st.session_state:
            st.session_state.key_value = str(random.randint(0, 1000000))

        if st.button("🔄 Tải lại nếu không thấy canvas", key="predict_reload_canvas"):
            st.session_state.key_value = str(random.randint(0, 1000000))

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=150,
            width=150,
            drawing_mode="freedraw",
            key=st.session_state.key_value,
            update_streamlit=True
        )
        if st.button("Dự đoán số từ bản vẽ", key="predict_from_drawing"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
                img = img.resize((28, 28)).convert("L")
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)
            else:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    else:
        uploaded_file = st.file_uploader("📤 Tải ảnh lên (định dạng PNG/JPG)", type=["png", "jpg", "jpeg"], key="predict_file_uploader")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh đã tải lên", width=150)
            if st.button("Dự đoán số từ ảnh", key="predict_from_upload"):
                img = Image.open(uploaded_file).convert("L")
                img = img.resize((28, 28))
                img = np.array(img, dtype=np.float32) / 255.0
                img = img.reshape(1, -1)

    if img is not None:
        st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)
        
        # Dự đoán với mô hình Keras
        probabilities = selected_model.predict(img)  # Keras trả về xác suất cho tất cả các lớp
        prediction = np.argmax(probabilities, axis=1)  # Lấy lớp có xác suất cao nhất
        st.subheader(f"🔢 Dự đoán: {prediction[0]}")

        # Tính độ tin cậy (xác suất của lớp được dự đoán)
        predicted_class_confidence = probabilities[0][prediction[0]]
        st.write(f"📈 **Độ tin cậy:** {predicted_class_confidence:.4f} ({predicted_class_confidence * 100:.2f}%)")

        # Hiển thị xác suất cho từng lớp
        st.write("**Xác suất cho từng lớp (0-9):**")
        confidence_df = pd.DataFrame({"Nhãn": range(10), "Xác suất": probabilities[0]})
        st.bar_chart(confidence_df.set_index("Nhãn"))
        show_experiment_selector()
# Tab MLflow
def show_experiment_selector():
    st.header("📊 MLflow Experiments")
    if 'mlflow_url' not in st.session_state:
        st.warning("⚠️ URL MLflow chưa được khởi tạo!")
        mlflow_input()

    st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
    experiment_name = "Neural_Network"
    
    try:
        # Kiểm tra experiment trong MLflow
        experiments = mlflow.search_experiments()
        selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

        if not selected_experiment:
            st.error(f"❌ Không tìm thấy Experiment '{experiment_name}'!", icon="🚫")
            return

        st.subheader(f"📌 Experiment: {experiment_name}")
        st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
        st.write(f"**Trạng thái:** {'🟢 Active' if selected_experiment.lifecycle_stage == 'active' else '🔴 Deleted'}")
        st.write(f"**Artifact Location:** `{selected_experiment.artifact_location}`")

        # Lấy tất cả các run từ MLflow
        runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])
        if runs.empty:
            st.warning("⚠ Không có runs nào trong experiment này!", icon="🚨")
            return

        st.subheader("🏃‍♂️ Danh sách Runs (Mô hình đã lưu trong MLflow)")
        run_info = []

        # Lọc các run có mô hình (kiểm tra artifact 'neural_network')
        client = mlflow.tracking.MlflowClient()
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_data = mlflow.get_run(run_id)
            run_name = run_data.info.run_name if run_data.info.run_name else f"Run_{run_id[:8]}"
            
            # Kiểm tra xem run có chứa mô hình không
            artifacts = client.list_artifacts(run_id)
            has_model = any(artifact.path.startswith("neural_network") for artifact in artifacts)
            
            if has_model:
                run_info.append((run_name, run_id))

        if not run_info:
            st.warning("⚠ Không tìm thấy mô hình nào trong các run của experiment này!", icon="🚨")
            return

        # Tạo danh sách run_name để chọn
        run_name_to_id = dict(run_info)
        run_names = list(run_name_to_id.keys())
        st.write("Danh sách run_name từ MLflow:", run_names)  # Debug

        # Chọn run từ danh sách
        selected_run_name = st.selectbox("🔍 Chọn Run:", run_names, key="run_selector")
        selected_run_id = run_name_to_id[selected_run_name]
        selected_run = mlflow.get_run(selected_run_id)

        if selected_run:
            st.markdown(f"<h3 style='color: #28B463;'>📌 Chi tiết Run: {selected_run_name}</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("#### ℹ️ Thông tin cơ bản")
                st.info(f"**Run Name:** {selected_run_name}")
                st.info(f"**Run ID:** `{selected_run_id}`")
                st.info(f"**Trạng thái:** {selected_run.info.status}")
                start_time_ms = selected_run.info.start_time
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
                st.info(f"**Thời gian chạy:** {start_time}")

            with col2:
                # Hiển thị Parameters (11)
                params = selected_run.data.params
                st.write("#### ⚙️ Parameters")
                if params:
                    st.write("- **Total Samples**: ", params.get("total_samples", "N/A"))
                    st.write("- **Test Size**: ", params.get("test_size", "N/A"))
                    st.write("- **Validation Size**: ", params.get("validation_size", "N/A"))
                    st.write("- **Train Size**: ", params.get("train_size", "N/A"))
                    st.write("- **K-Folds**: ", params.get("k_folds", "N/A"))
                    st.write("- **Number of Layers**: ", params.get("num_layers", "N/A"))
                    st.write("- **Neurons per Layer**: ", params.get("num_neurons", "N/A"))
                    st.write("- **Activation Function**: ", params.get("activation", "N/A"))
                    st.write("- **Optimizer**: ", params.get("optimizer", "N/A"))
                    st.write("- **Epochs**: ", params.get("epochs", "N/A"))
                    st.write("- **Learning Rate**: ", params.get("learning_rate", "N/A"))
                else:
                    st.warning("⚠ Không tìm thấy tham số nào cho run này!")

                # Hiển thị Metrics (4)
                metrics = selected_run.data.metrics
                st.write("#### 📊 Metrics")
                if metrics:
                    st.write("- **CV Accuracy Mean**: ", f"{metrics.get('cv_accuracy_mean', 'N/A'):.4f}")
                    st.write("- **CV Loss Mean**: ", f"{metrics.get('cv_loss_mean', 'N/A'):.4f}")
                    st.write("- **Test Accuracy**: ", f"{metrics.get('test_accuracy', 'N/A'):.4f}")
                    st.write("- **Test Loss**: ", f"{metrics.get('test_loss', 'N/A'):.4f}")
                else:
                    st.warning("⚠ Không tìm thấy chỉ số nào cho run này!")

    except Exception as e:
        st.error(f"❌ Lỗi khi truy cập MLflow: {str(e)}")
        traceback.print_exc()
# Giao diện chính
def main():
    if "mlflow_initialized" not in st.session_state:
        mlflow_input()
        st.session_state.mlflow_initialized = True

    st.title("🚀 Neural Network Classification App")
    tab1, tab2, tab3, tab4,tab5 = st.tabs([
        "📘 Lý thuyết NEURAL NETWORK",
        "📊 Dữ liệu",
        "📊 Chia Dữ liệu",
        "🧠 Huấn luyện",
        "🖥️ Dự Đoán"
    ])

    with tab1:
        explain_nn()
    with tab2:
        data()
      
        
    with tab3:
        split_data()
        
    with tab4:
        train()
        
    with tab5:
        du_doan()


if __name__ == "__main__":
    main()