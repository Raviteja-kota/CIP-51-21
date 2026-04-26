import os
import glob

for filepath in glob.glob('templates/*.html'):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace('{% end block %}', '{% endblock %}')
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
print("Fixed templates.")
