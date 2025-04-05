const canvas = document.getElementById("gameBoard");
const ctx = canvas.getContext("2d");

const boardSize = 8;
const tileSize = canvas.width / boardSize;

let board = [];
let currentPlayer = "black";

function initBoard() {
  board = Array.from({ length: boardSize }, () =>
    Array(boardSize).fill(null)
  );

  // Khởi tạo 4 quân cờ ở giữa
  board[3][3] = "white";
  board[3][4] = "black";
  board[4][3] = "black";
  board[4][4] = "white";
}

function drawBoard() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Vẽ các ô
  for (let row = 0; row < boardSize; row++) {
    for (let col = 0; col < boardSize; col++) {
      ctx.strokeStyle = "black";
      ctx.strokeRect(col * tileSize, row * tileSize, tileSize, tileSize);

      const piece = board[row][col];
      if (piece) {
        drawPiece(col, row, piece);
      }
    }
  }
}

function drawPiece(col, row, color) {
  ctx.beginPath();
  ctx.arc(
    col * tileSize + tileSize / 2,
    row * tileSize + tileSize / 2,
    tileSize / 2 - 5,
    0,
    Math.PI * 2
  );
  ctx.fillStyle = color;
  ctx.fill();
  ctx.closePath();
}

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const col = Math.floor(x / tileSize);
  const row = Math.floor(y / tileSize);

  handleClick(row, col);
});

function handleClick(row, col) {
  if (board[row][col] !== null) return;

  // TODO: Kiểm tra hợp lệ & lật quân (phần logic nâng cao)
  board[row][col] = currentPlayer;
  currentPlayer = currentPlayer === "black" ? "white" : "black";
  document.getElementById("currentPlayer").textContent =
    currentPlayer === "black" ? "Đen" : "Trắng";
  drawBoard();
}

function startNewGame() {
  currentPlayer = "black";
  initBoard();
  drawBoard();
  document.getElementById("currentPlayer").textContent = "Đen";
}

startNewGame();
