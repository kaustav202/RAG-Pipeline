import gradio as gr


def wrapper_function(input_string, fn):
    r = fn.run(input_string)
    return r


interface = gr.Interface(
    fn=wrapper_function,      
    inputs=gr.Textbox(label="Input String"),  
    outputs=gr.Textbox(label="Processed String") 
)

interface.launch(share = True)
