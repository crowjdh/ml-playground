from visualizer import tensor_visualizer_server


def main():
    tensor_visualizer_server.app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
