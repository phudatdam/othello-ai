const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;

const renderBoard = (board, boardState) => {
    board.innerHTML = '';
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            // draw cell
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            board.appendChild(cell);
            // draw disc if there is a disc
            if (boardState[row][col] != EMPTY) {
                const disc = document.createElement('div');
                disc.className = 'disc';
                disc.classList.add(boardState[row][col] == BLACK ? 'black' : 'white');
                cell.appendChild(disc);
          }
        }
    }
}

// Display valid moves and make them clickable
const displayValidMoves = (board, validMoves) => {
    for (const [row, col] of validMoves) {
        const cell = board.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
        cell.dataset.valid = "true";
        const disc = document.createElement('div');
        disc.className = 'disc';
        disc.classList.add('valid-move');
        cell.appendChild(disc);
    }
};

// Update players' scores and turn
// Cập nhật điểm số và lượt chơi hiện tại
const updateStatus = (boardState, turn) => {
    let blackCount = 0;
    let whiteCount = 0;
  
    // Duyệt toàn bộ bàn cờ để đếm số quân
    for (let row = 0; row < boardState.length; row++) {
      for (let col = 0; col < boardState[row].length; col++) {
        const cell = boardState[row][col];
        if (cell === BLACK) {
          blackCount++;
        } else if (cell === WHITE) {
          whiteCount++;
        }
      }
    }
  
    // Cập nhật điểm lên giao diện HTML
    document.getElementById('black-score').textContent = blackCount;
    document.getElementById('white-score').textContent = whiteCount;
  
    // Cập nhật lượt chơi
    const turnDisplay = document.getElementById('current-turn');
    if (turn === BLACK) 
      turnDisplay.textContent = "Black's turn";
    else
      turnDisplay.textContent = "White's turn";
  };
  

export { EMPTY, BLACK, WHITE, renderBoard, displayValidMoves, updateStatus };
