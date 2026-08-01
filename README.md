# Full Stack MOSS App

This repository contains a full-stack EEG processing application with a Next.js frontend, a Rust API server, a Rust websocket server, and a TimescaleDB/Postgres database.

## Prerequisites

Before starting the app, make sure you have:

- Docker Desktop or Docker Engine with Compose enabled
- Internet access for the initial image and package downloads
- Ports 3000, 8080, 9000, and 5432 available on your machine

## Run the full stack with Docker Compose

From the repository root, run:

```sh
docker compose up --build
```

The first run can take 10–15 minutes while Docker builds the images and installs dependencies. After that, startup is much faster.

Once the services are up, open:

- Frontend: http://localhost:3000
- API: http://localhost:9000
- Websocket: ws://localhost:8080

### Services in the stack

- db: TimescaleDB/Postgres database
- api-server: Rust API service
- websocket-server: Rust websocket service for EEG data handling
- frontend: Next.js interface

## Useful commands

```sh
# Stop everything
docker compose down

# Rebuild and start again
docker compose up --build

# View logs for one service
docker compose logs -f api-server
```

## Troubleshooting

### Build errors while installing Python packages

If you see errors related to NumPy, SciPy, or Torch during the container build, the Dockerfiles have been updated to use a compatible dependency set. If you still see issues, make sure Docker is using a recent version and that the machine has internet access.

### Port conflicts

If one of the ports is already in use, stop the conflicting process or change the port mapping in [docker-compose.yml](docker-compose.yml).

### Database readiness

The API and websocket services depend on the database. If they fail to connect at first, wait for the database healthcheck to pass and then restart the dependent containers.

## Local development (optional)

### Frontend

```sh
cd frontend
npm install
npm run dev
```

You can also run the mock signal server separately:

```sh
node server.js
```

### Backend

The backend can also be run locally if you want to debug Rust services directly.

#### Database setup

Start the database container separately if needed:

```sh
docker compose up -d db
```

Set the database URL for local Rust tooling:

```powershell
$env:DATABASE_URL="postgres://postgres:my_secure_password_123@localhost:5432/postgres"
```

Run the Rust migrations from the backend workspace:

```sh
cd backend
sqlx migrate run
```

Run the API server:

```sh
$env:RUST_LOG="info"
cd backend/api-server
cargo run
```

Run the websocket server:

```sh
$env:RUST_LOG="info"
cd backend/websocket-server
cargo run
```

### LSL setup

The LSL integration uses native build tools, so make sure CMake and a C/C++ compiler are installed:

```sh
cmake --version
```

If you need the Muse tools for a live headset stream:

```sh
pip install muselsl
```

To inspect connected Muse devices:

```sh
muselsl list
```

To start a stream from the first device:

```sh
muselsl stream
```

