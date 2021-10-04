from tkinter import filedialog

file_paths = filedialog.askopenfilenames(initialdir="./data", title="Select tagged text files",
                                         filetypes=(("tagged files", "*.tt .vert .tag .txt"), ("all files", "*.*")))

for file in file_paths:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # All lines including the blank ones
        lines.reverse()

        with open(file, 'w', encoding='utf-8') as g:
            g.write(''.join(lines))
