# game.py
from numpy import character
import pygame
import random
import sys
import numpy as np
from block import CharacterBlock, EmptyBlock, EnemyBlock

np.set_printoptions(threshold=sys.maxsize)

NUM_CHANNELS = 1
NUM_ACTIONS = 4

class GameTransition:
    
    def __init__(self, 
                 screen_width,
                 screen_height,
                 character, enemy,
                 c_speed, e_speed,
                 c_width, e_width,
                 c_height, e_height,
                 character_pos, enemy_pos,
                 ):
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.character = character # charcater
        self.enemy = enemy         # enemy     
        self.character_speed = c_speed
        self.enemy_speed = e_speed
        self.character_width = c_width
        self.enemy_width = e_width
        self.character_height = c_height
        self.enemy_height = e_height
        self.character_pos = character_pos # 0:x, 1:y
        self.enemy_pos = enemy_pos         # 0:x, 1:y        
     
        #이동할 좌표
        self.to_x = 0
        self.to_y = 0
        
        # FPS를 위한 Clock 생성
        self.clock = pygame.time.Clock()
        self.dt = self.clock.tick(60)
        
    def fill_block(background_x, background_y,
               width, height,
               pos):
    
        background = np.zeros((background_x, background_y))
        # print (f"background:\n{background}")


        print(pos[0])
        for x in range(width):
            for y in range(height):
                background[pos[0]-int(width/2)+x][pos[1]-int(height/2)+y] = 1
                
        # print (f"background:\n{background}")
      
    def _generate_character_enemy_(self) :
        empty_block = np.zeros((self.screen_height, self.screen_width))
        for i in range(self.screen_height):
            for j in range(self.screen_width):
                empty_block[i][j] = EmptyBlock.get_code()

    
        #character 
        if len(empty_block) > 0: 
           for x in range(self.character_width):
                for y in range(self.character_height):
                    x_poistion = self.character_pos[0]-int(self.character_width/2) 
                    y_position = self.character_pos[1]-int(self.character_height/2) 
                    empty_block[x_poistion+x][y_position+y] = CharacterBlock.get_code()
                    
        # enemy    
        if len(empty_block) > 0: 
           for x in range(self.enemy_width):
                for y in range(self.enemy_height):
                    x_poistion = self.enemy_pos[0]-int(self.enemy_width/2) 
                    y_position = self.enemy_pos[1]-int(self.enemy_height/2) 
                    empty_block[x_poistion+x][y_position+y] = EnemyBlock.get_code()
                
        # print(f"emptyblock:\n{empty_block}")
        return empty_block
                    
    def check_collision(self) -> bool :
        
        collision = False
        
        # 충돌 처리를 위한 rect 업데이트
        character_rect = self.character.get_rect()
        character_rect.left = self.character_pos[0]
        character_rect.top = self.character_pos[1]
        
        enmey_rect = self.enemy.get_rect()
        enmey_rect.left = self.enemy_pos[0]
        enmey_rect.top = self.enemy_pos[1]
        
        # 충돌 체크
        if character_rect.colliderect(enmey_rect):
            collision = True
            print("collision!")
            
        return collision
            
            
    def enemy_move(self):
        self.enemy_pos[1] += self.enemy_speed * self.dt

         # enemy 경계값 넘어가면 위치 초기화
        if self.enemy_pos[1] >= self.screen_height - self.enemy_height :
            self.enemy_pos[1] = 0
            self.enemy_pos[0] = random.randint(0, self.screen_width - self.enemy_width)
        
    
    def check_side(self):
         # x 경계값 처리
        if self.character_pos[0] < 0:
            self.character_pos[0] = 0
        elif self.character_pos[0] > self.screen_width - self.character_width:
            self.character_pos[0] = self.screen_width - self.character_width

        # y 경계값 처리
        if self.character_pos[1] < 0: 
            self.character_pos[1] = 0
        elif self.character_pos[1] > self.screen_height - self.character_height:
            self.character_pos[1] = self.screen_height - self.character_height
          
        # enemy x 경계값 처리
        if self.enemy_pos[0] < 0:
            self.enemy_pos[0] = 0
        elif self.enemy_pos[0] > self.screen_width - self.enemy_width:
            self.enemy_pos[0] = self.screen_width - self.enemy_width

        # enemy y 경계값 처리
        if self.enemy_pos[1] < 0: 
            self.enemy_pos[1] = 0
        elif self.enemy_pos[1] > self.screen_height - self.enemy_height:
            self.enemy_pos[1] = self.screen_height - self.enemy_height

       

    def move_right(self):
        
        # enemry 이동
        self.enemy_move()
        # 우측 이동
        self.character_pos[0] +=  self.character_speed * self.dt
        
        # 경계값처리
        self.check_side()
    
        # 충돌 체크
        if self.check_collision() is True:   
            return -1, True

        return 0, False
     
    def move_left(self):
        # enemry 이동
        self.enemy_move()
        
        # 좌측 이동
        self.character_pos[0] -=  self.character_speed * self.dt
        
        # 경계값처리
        self.check_side()
        
        if self.check_collision() is True:   
            return -1, True

        return 0, False
    
    def move_up(self):
        # enemry 이동
        self.enemy_move()
       
        # 위 이동
        self.character_pos[1] -=  self.character_speed * self.dt
        
        # 경계값처리
        self.check_side()
        
        if self.check_collision() is True:   
            return -1, True

        return 0, False
    
    def move_down(self):
        # enemry 이동
        self.enemy_move()
        
        # 아래 이동
        self.character_pos[1] +=  self.character_speed * self.dt
        
        # 경계값처리
        self.check_side()
        
        if self.check_collision() is True:   
            return 1, True

        return 0, False
    
    def get_state(self):
        
        field = self._generate_character_enemy_()
        # print(field)
        pos = np.array([self.character_pos,
                 self.enemy_pos])
        return field, pos

    
    
class AgentAction:
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3

class Game():
    ACTIONS = {
        AgentAction.MOVE_LEFT: 'move_left',
        AgentAction.MOVE_RIGHT: 'move_right',
        AgentAction.MOVE_UP: 'move_up',
        AgentAction.MOVE_DOWN: 'move_down',
    }
    
    def __init__(self, field_width, field_height):
        
        pygame.init()
        
        # init
        self.screen_width = field_width
        self.screen_height = field_height
        self.screen = pygame.display.set_mode((self.screen_width,
                                               self.screen_height))
         #background
        self.background = pygame.image.load("D:/tank_env/code/pygame_dev/images/background.png")
        #character 
        self.character = pygame.image.load("D:/tank_env/code/pygame_dev/images/character.png")
        #enemy 
        self.enemy = pygame.image.load("D:/tank_env/code/pygame_dev/images/enemy.png")
        
        self.charater_size = self.character.get_rect().size
        self.character_width = self.charater_size[0]
        self.character_height = self.charater_size[1]
        self.character_pos = np.array([0, 0])
        self.character_pos[0] = self.screen_width/2 - self.character_width/2
        self.character_pos[1] = self.screen_height - self.character_height
        self.character_pos_init = self.character_pos
        
        # character speed
        self.character_speed = 0.3

        self.enemy_size = self.enemy.get_rect().size
        self.enemy_width = self.enemy_size[0]
        self.enemy_height = self.enemy_size[1]
        self.enemy_pos = np.array([0, 0])
        self.enemy_pos[0] = random.randint(0, self.screen_width - self.enemy_width)
        self.enemy_pos[1] = 0
        self.enemy_pos_init = self.enemy_pos
        # enemy speed
        self.enemy_speed = 0.3

        # 폰트 정의
        game_font = pygame.font.Font(None, 40)

        # 총 시간
        total_time = 10

        # 시작 시간 정보
        start_ticks = pygame.time.get_ticks()
        



        self.reset()
        
    def reset(self):
        
        
        c_pos = np.array([random.randint(0, self.screen_width - self.enemy_width),
                 self.character_pos_init[1]])
        e_pos = np.array([random.randint(0, self.screen_width - self.enemy_width),
                 self.enemy_pos_init[1]])
        
        self.state_transition = GameTransition(self.screen_width,
                                               self.screen_height,
                                               self.character,
                                               self.enemy,
                                               c_speed = self.character_speed,
                                               e_speed = self.enemy_speed,
                                               c_width= self.character_width,
                                               e_width= self.enemy_width,
                                               c_height= self.character_width,
                                               e_height= self.enemy_width,
                                               character_pos = c_pos,
                                               enemy_pos = e_pos)
    
        self.tot_reward = 0
        field, pos = self.state_transition.get_state()
        return field
    
    def step(self, action):
        
        reward, done = getattr(self.state_transition, Game.ACTIONS[action])()
        self.tot_reward += reward
        field, pos = self.state_transition.get_state()
        return field, pos, reward, done
    
    
    def quit(self):
        pygame.time.delay(2000) # 2초 정도 대기 
        pygame.quit()
            
            
    def render(self, fps, pos) -> bool :

        
        pygame.display.set_caption("Game for RL")
        pygame.event.pump()
        
        
        # return self.env.get_length()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Done = True
                    
        # pump: 파이 게임을 시스템과 동기화 상태로 유지하려면 모든 것을 최신 상태로 유지
        #pygame.event.pump()
        # 신규 position
        
        self.character_pos = pos[0]
        self.enemy_pos = pos[1]
        
        
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.character, (self.character_pos[0], self.character_pos[1]))
        self.screen.blit(self.enemy, (self.enemy_pos[0], self.enemy_pos[1]))
        
        pygame.display.flip()

