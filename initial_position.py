import numpy as np
import os


def initialize_position(n, a):
  b0 = np.array([a, 0., 0.])
  b1 = np.array([a / 2, (a * np.sqrt(3)) / 2., 0.])
  b2 = np.array([a / 2, (a * np.sqrt(3)) / 6., a * np.sqrt(2/3)])

  r = np.zeros((n**3, 3))

  for i_0 in range(n):
      for i_1 in range(n):
          for i_2 in range(n):
              i = i_0 + i_1*n + i_2*n*n
              pos_of_one_particle = (i_0 - (n - 1) / 2) * b0 + (i_1 - (n - 1) / 2) * b1 + (
                      i_2 - (n - 1) / 2) * b2
              r[i] = pos_of_one_particle
  return r
  


def get_positions_from_xyz_file(filename):
  positions = []
  with open(filename, "r") as infile:
    lines = infile.readlines()
    lines.pop(0)
    lines.pop(0)
    for line in lines:
      splitted_line = line.split(' ')
      position = []
      for item in splitted_line:
        try:
          x = float(item)
          position.append(x)
        except:
          pass
      positions.append(position)
  return np.array(positions)


def create_initial_pos_file_in_xyz_format(n, a, create_new_initial_file=False):
  # Checks if "initial_file.xyz" already exist and if is in a proper xyz file for n**3 particles 
  filename = "initial_file.xyz"
  N = 0
  try:
    with open(filename) as file:
      try:
        line = file.readline()
        try:
          N = int(line)
        except:
          print(f"First line of ,,{filename}'' does not contain an integer!")
      except:
        print(f"File ,,{filename}'' is empty")
  except:
    print(f"No such file as ,,{filename}''")

  # Crete new xyz file if needed
  if N != n**3 or create_new_initial_file: 
    positions = initialize_position(n, a)
    # Write positions to proper xyz initial file
    with open(filename, "w") as outfile:
        N, _ = positions.shape
        outfile.write(str(N) + '\n' + '\n')
        for vector in positions:
            outfile.write(f'Ar {vector[0]} {vector[1]} {vector[2]} \n')
    print(f"File ,,{filename}'' created.")
  else:
    print(f",,{filename}'' already exist.")

  return get_positions_from_xyz_file(filename="initial_file.xyz")
