import chess
import pygame
import sys
import math
from functools import lru_cache
import time
import chess.svg
import cairosvg
import io
import torch
from train import ChessNet  
import numpy as np
import os
pygame.init()


SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE)
FPS = 60


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT = (255, 255, 0, 50)


material_weights = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}


PIECE_SQUARE_TABLES = {
    'P': [  # Pawn
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ]
}

screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Chess AI")
clock = pygame.time.Clock()

def render_board(board):
    svg_data = chess.svg.board(board=board, size=BOARD_SIZE).encode('utf-8')
    png_data = cairosvg.svg2png(bytestring=svg_data)
    image = pygame.image.load(io.BytesIO(png_data))
    return pygame.transform.scale(image, (BOARD_SIZE, BOARD_SIZE))

def draw_board(screen, board):
    board_image = render_board(board)
    screen.blit(board_image, (0, 0))

def highlight_square(screen, square):
    x = chess.square_file(square) * SQUARE_SIZE
    y = (7 - chess.square_rank(square)) * SQUARE_SIZE
    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surface.fill(HIGHLIGHT)
    screen.blit(highlight_surface, (x, y))

def get_square_from_pos(pos):
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)

def get_board_matrix(board):
    matrix = np.zeros((8, 8, 12), dtype=np.float32)
    piece_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_type = str(piece)
            matrix[rank][file][piece_idx[piece_type]] = 1
            
    
    mobility = np.zeros((8, 8, 1), dtype=np.float32)
    protection = np.zeros((8, 8, 1), dtype=np.float32)
    attacks = np.zeros((8, 8, 1), dtype=np.float32)
    
    
    matrix = np.concatenate([matrix, mobility, protection, attacks], axis=2)
    matrix = matrix.transpose(2, 0, 1)
    return matrix

def evaluate_position(board):
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            if not piece.color:
                rank = 7 - rank
            if piece.symbol().upper() in PIECE_SQUARE_TABLES:
                score += PIECE_SQUARE_TABLES[piece.symbol().upper()][rank][file]
    return score

def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0

    
    material_eval = sum(material_weights[p.piece_type] * (1 if p.color == chess.WHITE else -1)
                       for p in board.piece_map().values())
    
   
    position_eval = evaluate_position(board)
    
    # Center control
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center_control = sum(10 if board.is_attacked_by(chess.WHITE, sq) else
                        -10 if board.is_attacked_by(chess.BLACK, sq) else 0
                        for sq in center_squares)
    
    
    total_eval = material_eval + position_eval * 0.1 + center_control
    
    return total_eval if board.turn else -total_eval

@lru_cache(maxsize=None)
def cached_evaluate_board(fen):
    return evaluate_board(chess.Board(fen))

def order_moves(board, moves):
    return sorted(moves, key=lambda move: (board.is_capture(move), board.gives_check(move)), reverse=True)

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return cached_evaluate_board(board.fen())

    moves = order_moves(board, board.legal_moves)
    if maximizing_player:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_ai_move(board, depth=4):
    try:
       
        model = ChessNet()
        model.load_state_dict(torch.load('chess_model_best.pth'))
        model.eval()  
        use_neural_net = True
        print("Using neural network for evaluation")
    except Exception as e:
        print(f"Neural network model not found: {e}")
        print("Using traditional evaluation only")
        use_neural_net = False
    
    start_time = time.time()
    legal_moves = list(board.legal_moves)
    move_evaluations = []
    
    for move in legal_moves:
        board.push(move)
        
        
        traditional_eval = minimax(board, depth - 1, -math.inf, math.inf, False)
        
        if use_neural_net:
            
            board_matrix = get_board_matrix(board)
            
            board_tensor = torch.FloatTensor(board_matrix).unsqueeze(0)
            with torch.no_grad():
                nn_prediction = model(board_tensor).item()
            
            combined_eval = 0.7 * traditional_eval + 0.3 * (nn_prediction * 2000 - 1000)
        else:
            combined_eval = traditional_eval
        
        move_evaluations.append((move, combined_eval))
        board.pop()
    
    
    top_moves = sorted(move_evaluations, key=lambda x: x[1], reverse=True)[:3]
    best_move = np.random.choice([move for move, _ in top_moves], 
                                p=[0.7, 0.2, 0.1] if len(top_moves) == 3 else None)
    
    end_time = time.time()
    print(f"AI move: {board.san(best_move)} ({end_time - start_time:.2f}s)")
    
    return best_move

def main():
    board = chess.Board()
    selected_square = None
    move_in_progress = False
    difficulty = 4  

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE:
                pos = pygame.mouse.get_pos()
                clicked_square = get_square_from_pos(pos)

                if selected_square is None:
                    selected_square = clicked_square
                else:
                    move = chess.Move(selected_square, clicked_square)
                    if move in board.legal_moves:
                        move_in_progress = True
                        board.push(move)
                        selected_square = None
                    else:
                        selected_square = clicked_square

        screen.fill(WHITE)
        draw_board(screen, board)

        if selected_square is not None:
            highlight_square(screen, selected_square)

        pygame.display.flip()
        clock.tick(FPS)

        if move_in_progress:
            pygame.time.wait(500) 
            print("AI is thinking...")
            ai_move = get_ai_move(board, depth=difficulty)
            board.push(ai_move)
            move_in_progress = False

        if board.is_game_over():
            print("Game Over!")
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                print(f"Checkmate! {winner} wins!")
            elif board.is_stalemate():
                print("Stalemate!")
            elif board.is_insufficient_material():
                print("Draw due to insufficient material!")
            elif board.can_claim_fifty_moves():
                print("Draw by fifty-move rule!")
            elif board.can_claim_threefold_repetition():
                print("Draw by threefold repetition!")
            else:
                print(f"Game result: {board.result()}")
            pygame.time.wait(5000)  
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    main()