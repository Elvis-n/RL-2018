# Plot norm
norm_values = read.csv('norm_values.csv', header = F)
norm_values$X = 1:nrow(norm_values)
names(norm_values) = c('Norm', 'Episodes')
plot(norm_values$Episodes,norm_values$Norm, xlab = 'Episodes',
     ylab='Norm', main = 'Norm vs. Episodes', type='l')


# Plot average Q
avg_values = read.csv('avg_values.csv', header = F)
avg_values$X = 1:nrow(avg_values)
names(avg_values) = c('Avg', 'Episodes')
plot(avg_values$Episodes, avg_values$Avg, xlab = 'Episodes',
     ylab='Average of Q', main = 'Q mean vs. Episodes', type='l')

get_q_action <- function(index, q, s) {
  return(which.max(c(q[index,1], q[index+s,1], q[index+2*s,1]))-1)
}

# Q plot
q_vals = read.csv('q_values.csv', header = F)
x_min = -1.2
x_max = .6
v_min = -.07
v_max = .07
x_size = 10
v_size = 30
action_size = 3
x_vec = seq(x_min, x_max, length.out = x_size)
v_vec = seq(v_min, v_max, length.out = v_size)
q_df = data.frame(matrix(0, x_size*v_size, action_size))
index = 1
for (x in x_vec) {
  for (v in v_vec) {
    a <- get_q_action(index,q_vals,x_size*v_size)
    q_df[index,] <- c(x,v,a)
    index <- index + 1
  }
}
names(q_df) <- c('Position', 'Velocity', 'Action')
plot(q_df$X1, q_df$X2, col=q_df$X3, xlab='Position',ylab='Velocity')
legend('topleft',legend=c('Back','Neutral','Forward'))

q_df$Action <- as.factor(as.character(q_df$Action))
library(ggplot2)
qplot(Position, Velocity, colour=Action, data=q_df)
