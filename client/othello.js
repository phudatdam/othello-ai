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
const displayValidMoves = (board, validMoves, turn) => {
    for (const [row, col] of validMoves) {
        const cell = board.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
        const disc = document.createElement('div');
        disc.className = 'disc';
        if (turn === BLACK) {
            cell.dataset.valid = "true";
            disc.classList.add('valid-move'); // green
        } else {
            disc.classList.add('opponent-move'); // red
        }
        cell.appendChild(disc);
    }
};

// Update players' scores and turn
const updateStatus = (boardState, turn) => {
    const blackScore = boardState.flat().filter(cell => cell === BLACK).length;
    const whiteScore = boardState.flat().filter(cell => cell === WHITE).length;

    document.getElementById("black-score").textContent = `${blackScore} ⚫`;
    document.getElementById("white-score").textContent = `⚪ ${whiteScore}`;

    const turnText = turn === BLACK ? "Black's turn" : "White's turn";
    document.getElementById("turn-indicator").textContent = turnText;
}

export { EMPTY, BLACK, WHITE, renderBoard, displayValidMoves, updateStatus };
