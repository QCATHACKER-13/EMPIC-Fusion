import turtle
import random as r

adrian = turtle.Screen()
adrian.bgcolor(54,54,54)

bert = turtle.Turtle()
bert.shape("turtle")
bert.color(0,200,0)
bert.penup()
# bert.speed(0)
bert.goto(-150,150)

# Original speed for particles.
origSpeed = 0

# List/array
berts = []

# Make list of turtles!
for i in range(4):
  temp = turtle.Turtle()
  temp.shape("circle")
  temp.color(0,200,0)
  temp.pencolor(255,255,255)
  temp.speed(origSpeed)
  temp.penup()
  temp.velx = 0
  temp.vely = 0
  temp.accx = r.randint(-10,10)
  temp.accy = r.randint(-10,10)
  berts.append(temp)

def newPart(x,y):
  temp = turtle.Turtle()
  temp.setx(x)
  temp.sety(y)
  temp.shape("circle")
  temp.color(0,200,0)
  temp.pencolor(255,255,255)
  temp.speed(origSpeed)
  # temp.penup()
  temp.velx = 0
  temp.vely = 0
  temp.accx = r.randint(-10,10)
  temp.accy = r.randint(-10,10)
  
  berts.append(temp)

for turt in berts:
  x = r.randint(-150,150)
  y = r.randint(-150,150)
  turt.goto(x,y)


def bounce(turt):
  myY = turt.ycor()
  myX = turt.xcor()
  edge = 150
  # Bounce of x edges.
  if myX < -edge or myX > edge:
    turt.velx *= -1
    shade = r.randint(72,255)
    turt.color(shade,0,shade)
    turt.pencolor(255,255,255)
  # Bounce of y edges.
  if myY < -edge or myY > edge:
    turt.vely *= -1
    shade = r.randint(72,255)
    turt.color(0,shade,0)
    turt.pencolor(255,255,255)
  

  
# Pac-man wrap-around.
def pacWrap(turt):
  global origSpeed
  myY = turt.ycor()
  myX = turt.xcor()
  edge = 150
  # Check X bound.
  if myX < -edge:
    turt.hideturtle()
    turt.speed(0)
    turt.goto(edge, myY)
    turt.showturtle()
  elif myX > edge:
    turt.hideturtle()
    turt.speed(0)
    turt.goto(-edge, myY)
    turt.showturtle()
  # Check Y bound.
  if myY < -edge:
    turt.hideturtle()
    turt.speed(0)
    turt.goto(myX, edge)
    turt.showturtle()
  elif myY > edge:
    turt.hideturtle()
    turt.speed(0)
    turt.goto(myX, -edge)
    turt.showturtle()
  turt.speed(origSpeed)
  
# Euler physics system :D
def physics(turt):
  turt.velx += turt.accx
  turt.vely += turt.accy
  turt.goto(turt.xcor() + turt.velx,
            turt.ycor() + turt.vely)
  # turt.setx(turt.xcor() + turt.velx)
  # turt.sety(turt.ycor() + turt.vely)
  
  # This is the friction!
  if abs(turt.velx) > 0:
    turt.velx *= 0.98
  if abs(turt.vely) > 0:
    turt.vely *= 0.98
  
  turt.accx = 0
  turt.accy = 0

# Add random force to all
# particles on the berts list.
def explode(mX,mY):
  for turt in berts:
    turt.accx = r.randint(-5,5)
    turt.accy = r.randint(-5,5)
    
adrian.onclick(explode) 
adrian.onclick(newPart)
  
def collision(turt):
  for other in berts:
    # Make sure we are not
    # checking a particle
    # against itself
    if turt != other:
      tX = turt.xcor()
      tY = turt.ycor()
      oX = other.xcor()
      oY = other.ycor()
      dist = abs(tX-oX) + abs(tY-oY)
      if dist < 40:
        return True

def reverseV(turt):
  turt.velx *= -1
  turt.vely *= -1
  
def repel(turt):
  turt.velx = 0
  turt.vely = 0
  turt.accx = r.randint(-5,5)
  turt.accy = r.randint(-5,5)
  
running = True
adrian.listen()

# Main loop.
while running:
  for turt in berts:
    physics(turt)
    # pacWrap(turt)
    bounce(turt)
    if collision(turt):
      # Collision!
      reverseV(turt)
      # repel(turt)
  
  

  
