# Federated Learning Privacy Evaluation

This project explores privacy vulnerabilities in Federated Learning, such as Membership Inference Attacks (MIAs), and implements defense mechanisms like Differential Privacy and Dropout.

The system includes:
- A **React** frontend to configure and visualize training
- A **FastAPI** backend to manage the federated learning pipeline
- A **Dockerized** environment for consistent and reproducible deployment

---

## 🚀 Launch Instructions

To run this project locally:

1. **Start Docker Desktop**

   Make sure Docker Desktop is running on your machine.

2. **Open a terminal in the root directory of the project**

   If you're not already there:
   ```bash
   cd /path/to/your/project
   ```

3. **Launch the services using Docker Compose**
   ```bash
   docker compose up --build
   ```

4. **Access the services:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend (API): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Technologies Used

- **Federated Learning** with PyTorch
- **FastAPI** for backend APIs
- **React + Vite** for the frontend
- **Docker + Docker Compose** for deployment
- **GitLab CI/CD** with local runner (optional)

---

## 📌 Notes

- You must have Docker installed and running.
- The first build may take some time as it pulls dependencies.
- You can stop the containers with:
  ```bash
  docker compose down
  ```

---