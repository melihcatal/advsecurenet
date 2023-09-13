import click

@click.group()
@click.option('--name', default='World', help='The person to greet.')
def main():
    click.echo("Hello World!")

if __name__ == '__main__':
    main()
