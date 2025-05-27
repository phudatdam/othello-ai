import { BLACK, WHITE, renderBoard, displayValidMoves, updateStatus } from "./othello.js";

function initGame(websocket) {
    websocket.addEventListener("open", () => {
        // Send an "init" event according to who is connecting.
        const params = new URLSearchParams(window.location.search);
        let event = { type: "init" };
        if (params.has("join")) {
            // Second player joins an existing game.
            event.join = params.get("join");
        } else if (params.has("watch")) {
            // Spectator watches an existing game.
            event.watch = params.get("watch");
        } else if (params.has("solo")) {
            // Player chooses to play against the AI bot
            event.bot = true;
        } else {
            // First player starts a new game.
        }
        websocket.send(JSON.stringify(event));
    });
}

function showMessage(message) {
    window.setTimeout(() => window.alert(message), 50);
}

function receiveMessages(board, websocket) {
    websocket.addEventListener("message", ({ data }) => {
        const event = JSON.parse(data);
        switch (event.type) {
            case "init":
                // Create links for inviting the second player and spectators.
                document.querySelector(".join").href = "?join=" + event.join;
                document.querySelector(".watch").href = "?watch=" + event.watch;
                console.log("Join link:", "?join=" + event.join);
                console.log("Watch link:", "?watch=" + event.watch);
                // Highlight valid moves.
                displayValidMoves(board, event.validMoves, BLACK);
                break;
            case "play":
                // Update the board.
                renderBoard(board, event.boardState);
                // Highlight valid moves.
                displayValidMoves(board, event.validMoves, event.turn);
                // Update players' scores and turn.
                updateStatus(event.boardState, event.turn);
                break;
            case "win":
                if (!event.player) {
                    showMessage(`It's a draw!`);
                } else {
                    showMessage(`${event.player == BLACK ? 'Black' : 'White'} wins!`);
                }
                // No further messages are expected; close the WebSocket connection.
                websocket.close(1000);
                break;
            case "error":
                showMessage(event.message);
                break;
            default:
                throw new Error(`Unsupported event type: ${event.type}.`);
        }
    });
}

function sendMoves(board, websocket) {
    // Don't send moves for a spectator watching a game.
    const params = new URLSearchParams(window.location.search);
    if (params.has("watch")) {
        return;
    }

    // When clicking a cell that has class valid-move, send a "play" event for a move in that cell.
    board.addEventListener("click", ({ target }) => {
        target = target.closest(".cell");
        if (target.dataset.valid) {
            const event = {
                type: "play",
                row: parseInt(target.dataset.row, 10),
                col: parseInt(target.dataset.col, 10),
            };
            websocket.send(JSON.stringify(event));
        }
    });
}

window.addEventListener("DOMContentLoaded", () => {
    // Initialize the UI.
    const board = document.querySelector(".board");
    const boardState = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ];
    renderBoard(board, boardState);
    // Open the WebSocket connection and register event handlers.
    const websocket = new WebSocket("ws://localhost:8001/");
    initGame(websocket);
    receiveMessages(board, websocket);
    sendMoves(board, websocket);
});