import pygame
import pickle
import numpy as np
from ButtonLib import Button
from Identify import Identify
import torch


file = "DDQN_Breakout_identify_data.pkl"

with open(file, 'rb') as f:
    data = pickle.load(f)

pygame.init()
pygame.mixer.init()  # initialises pygame and sound
pygame.font.init()
all_fonts = pygame.font.get_fonts()
fontNumber = 4
myfont = pygame.font.SysFont(all_fonts[fontNumber], 25)
myFontLarge = pygame.font.SysFont(all_fonts[fontNumber], 40)


fps = 30  # the game's frames per second

info = pygame.display.Info()
monitorx = 1920
monitory = 1080
dispx, dispy = 1280, 720
if dispx > monitorx:  # scales screen down if too long
    dispy /= dispx / monitorx
    dispx = monitorx
if dispy > monitory:  # scales screen down if too tall
    dispx /= dispy / monitory
    dispy = monitory

dispx = int(dispx)  # So the resolution does not contain decimals
dispy = int(dispy)

screen = pygame.display.set_mode((dispx, dispy))

pygame.display.set_caption("RL-identifier")
clock = pygame.time.Clock()


reduceFrameButton = Button(120,5,60,60,(150,150,150),(70,70,70),True,(0,0,0),"<")
increaseFrameButton = Button(600,5,60,60,(150,150,150),(70,70,70),True,(0,0,0),">")
reduceFrameButton2 = Button(50,5,60,60,(150,150,150),(70,70,70),True,(0,0,0),"<<")
increaseFrameButton2 = Button(670,5,60,60,(150,150,150),(70,70,70),True,(0,0,0),">>")

reduceBatchButton = Button(940,70,60,60,(150,150,150),(70,70,70),True,(0,0,0),"<")
increaseBatchButton = Button(1200,70,60,60,(150,150,150),(70,70,70),True,(0,0,0),">")

modeButton = Button(1000,5,200,60,(150,150,150),(70,70,70),True,(0,0,0),"Change Mode")

running = True
cur_screen = "home"
mode = "env" # env or batch
frame = 0
batch_idx = 0
n = 10
discount = 0.99

max_frames = len(data.states)

while running:
    clock.tick(fps)

    # reset screen
    screen.fill((0, 0, 0))

    # event handling
    mouseUp = False
    pos = pygame.mouse.get_pos()
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            exit()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouseUp = True

    modeButton.create(screen)
    if modeButton.click(pos, mouseUp):
        if mode == "env":
            mode = "batch"
        else:
            mode = "env"

    if cur_screen == "home":
        reduceFrameButton.create(screen)
        if reduceFrameButton.click(pos, mouseUp) and frame > 0:
            frame -= 1

        increaseFrameButton.create(screen)
        if increaseFrameButton.click(pos, mouseUp) and frame < max_frames - 10:
            frame += 1

        reduceFrameButton2.create(screen)
        if reduceFrameButton2.click(pos, mouseUp) and frame > 100:
            frame -= 100

        increaseFrameButton2.create(screen)
        if increaseFrameButton2.click(pos, mouseUp) and frame < max_frames - 100 - 10:
            frame += 100

        textSurface = myFontLarge.render("Current Frame: " + str(frame), False, (255, 255, 255), 30)
        screen.blit(textSurface, (220, 20))


        if mode == "batch":

            textSurface = myfont.render("Batch Index: " + str(batch_idx), False, (255, 255, 255), 30)
            screen.blit(textSurface, (1030, 80))

            reduceBatchButton.create(screen)
            if reduceBatchButton.click(pos, mouseUp) and batch_idx > 0:
                batch_idx -= 1

            increaseBatchButton.create(screen)
            if increaseBatchButton.click(pos, mouseUp) and batch_idx < 31:
                batch_idx += 1

            textSurface = myfont.render("Updated Batch Q-Vals: " + str(np.around(data.batch_new_Qvals[frame][batch_idx], decimals=3)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 80))

            textSurface = myfont.render("Batch Q-Vals: " + str(np.around(data.batch_Qvals[frame][batch_idx], decimals=3)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 120))

            textSurface = myfont.render("Action Taken: " + str(data.er_actions[data.batch_idxs[frame]][batch_idx]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 160))

            textSurface = myfont.render("Batch Target State Values: " + str(np.around(data.batch_target_states_vals[frame][batch_idx], decimals=3)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 240))

            textSurface = myfont.render("Batch Target Next Max Action: " + str(data.batch_target_actions[frame][batch_idx]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 280))

            rewards = np.zeros(1, dtype=np.float32)
            terminals = np.zeros(1, dtype=np.bool_)
            for i in range(n):
                reward_batch = data.er_rewards[(data.batch_idxs[frame][batch_idx] + i)]
                if terminals:
                    reward_batch = 0.0
                rewards += reward_batch * (discount ** i)

                terminals = np.logical_or(terminals, data.er_dones[(data.batch_idxs[frame][batch_idx] + i)])

            textSurface = myfont.render("Batch Rewards: " + str(rewards[0]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 200))

            textSurface = myfont.render("Terminal? " + str(terminals[0]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 320))

            textSurface = myfont.render("Total Batch Loss " + str(np.around(data.batch_loss[frame], decimals=3)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 360))

            textSurface = myfont.render("Total Batch Policy Churn " + str(np.around(data.churn[frame], decimals=4)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 400))

            stacked_img = np.stack((data.er_states[data.batch_idxs[frame]][batch_idx][3],) * 3, axis=-1)
            surf = pygame.transform.rotate(pygame.transform.scale(pygame.surfarray.make_surface(stacked_img), (336, 336)), -90)
            screen.blit(surf, (500, 380))

            stacked_img = np.stack((data.er_next_states[(data.batch_idxs[frame][batch_idx] + n - 1)][3],) * 3, axis=-1)

            surf = pygame.transform.rotate(pygame.transform.scale(pygame.surfarray.make_surface(stacked_img), (336, 336)), -90)
            screen.blit(surf, (900, 380))

        else:
            textSurface = myfont.render("Q-vals: " + str(np.around(data.Qvals[frame], decimals=3)), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 120))

            stacked_img = np.stack((data.states[frame][3],) * 3, axis=-1)
            surf = pygame.transform.rotate(pygame.transform.scale(pygame.surfarray.make_surface(stacked_img), (336, 336)), -90)
            screen.blit(surf, (300, 300))

            textSurface = myfont.render("Action Taken: " + str(data.actions[frame]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 160))

            textSurface = myfont.render("Reward Received: " + str(data.rewards[frame]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 200))

            textSurface = myfont.render("Terminal? " + str(data.dones[frame]), False, (255, 255, 255), 30)
            screen.blit(textSurface, (20, 240))




    pygame.display.flip()

pygame.quit()