import json

def generate_upload_html():
    with open('template.html', 'r') as f:
        template = f.read()
    
    with open('style.css', 'r') as f:
        styles = f.read()
    
    with open('app.js', 'r') as f:
        scripts = f.read()
    
    html = template.replace('<!-- STYLES -->', styles).replace('<!-- SCRIPTS -->', scripts)
    
    with open('index.html', 'w') as f:
        f.write(html)

if __name__ == "__main__":
    generate_upload_html()
    print("Beautiful HTML UI generated: index.html")
